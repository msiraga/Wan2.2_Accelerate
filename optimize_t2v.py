# H200 defaults (one-time, e.g. in __init__ or before generate)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
PARAM_DTYPE = getattr(config, "param_dtype", torch.bfloat16)  # bf16 on H200

def _maybe_compile(fn, enable=True):
    if not enable:
        return fn
    try:
        return torch.compile(fn, mode="reduce-overhead", dynamic=True)
    except Exception:
        return fn

# ---- assume we already loaded and .to(device) the following on GPU:
# text_encoder, vae, low_model, high_model
# and we have: device, sp_size, patch_size, vae_stride, num_train_timesteps, boundary

# Small compiled call wrappers (safer than compiling full Module)
_f_low  = _maybe_compile(lambda x, t, **kw:  low_model(x,  t=t, **kw)[0],  enable=True)
_f_high = _maybe_compile(lambda x, t, **kw: high_model(x, t=t, **kw)[0],  enable=True)

# ----- shapes -----
zdim = vae.model.zdim
width, height = size
t_stride, h_stride, w_stride = config.vaestride if hasattr(config, "vaestride") else config.vae_stride
pt, ph, pw = config.patchsize if hasattr(config, "patchsize") else config.patch_size

T_out = (frame_num - 1) // t_stride + 1
H_out = height // h_stride
W_out = width  // w_stride
latent_shape = (zdim, T_out, H_out, W_out)

# sequence length (token count) aligned for SP if used
npt = math.ceil(T_out / pt); nph = math.ceil(H_out / ph); npw = math.ceil(W_out / pw)
seq_len = npt * nph * npw
if sp_size and sp_size > 1:
    seq_len = math.ceil(seq_len / sp_size) * sp_size

# ----- text enc -----
neg = n_prompt or getattr(config, "samplenegprompt", "")
ctx      = text_encoder([input_prompt], device)  # keep on GPU
ctx_null = text_encoder([neg],          device)

# ----- noise / latents -----
g = torch.Generator(device=device)
g.manual_seed(seed if seed >= 0 else random.randint(0, 2**31 - 1))
latents = torch.empty(latent_shape, dtype=torch.float32, device=device).normal_(generator=g)

# ----- scheduler / timesteps -----
boundary_t = float(getattr(config, "numtraintimesteps", getattr(config, "num_train_timesteps"))) * float(config.boundary)

if sample_solver.lower() == "unipc":
    sched = FlowUniPCMultistepScheduler(num_train_timesteps=num_train_timesteps, shift=1, use_dynamic_shifting=False)
    sched.set_timesteps(sampling_steps, device=device, shift=shift)
    timesteps = sched.timesteps
elif sample_solver.lower() in {"dpm++", "dpmpp"}:
    sched = FlowDPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, shift=1, use_dynamic_shifting=False)
    sigmas = getsamplingsigmas(sampling_steps, shift)
    timesteps, = retrieve_timesteps(sched, device=device, sigmas=sigmas)
else:
    raise NotImplementedError(f"Unsupported solver: {sample_solver}")

# fused CFG helper: single forward with batch=2 (uncond, cond)
def _fused_forward(net, x, t_scalar, ctx_u, ctx_c):
    # model expects (B, C, T, H, W) so add batch and stack
    xb = torch.stack([x, x], dim=0)          # (2, C, T, H, W)
    tb = torch.tensor([t_scalar, t_scalar], device=device)
    kwargs = {"context": [ctx_u, ctx_c], "seq_len": seq_len}  # many DiTs accept list/packed context per batch
    eps_b = net(xb, tb, **kwargs)            # (2, C, T, H, W)
    eps_u, eps_c = eps_b[0], eps_b[1]
    return eps_u, eps_c

# FSDP no_sync compat
@contextmanager
def _noop(): yield
nosync_low  = getattr(low_model,  "no_sync", getattr(low_model,  "nosync",  _noop))
nosync_high = getattr(high_model, "no_sync", getattr(high_model, "nosync", _noop))

guide_scale = (guide_scale, guide_scale) if isinstance(guide_scale, float) else guide_scale

with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=PARAM_DTYPE), nosync_low(), nosync_high():
    for t in tqdm(timesteps):
        t_val = float(t.item())
        use_high = (t_val >= boundary_t)
        net = _f_high if use_high else _f_low
        gscale = guide_scale[1] if use_high else guide_scale[0]

        # one forward pass for CFG
        eps_u, eps_c = _fused_forward(net, latents, t_val, ctx_null, ctx)
        eps = eps_u + gscale * (eps_c - eps_u)

        # scheduler step (add/remove batch dim once)
        x_in = latents.unsqueeze(0)      # (1, C, T, H, W)
        eps_b = eps.unsqueeze(0)         # (1, C, T, H, W)
        x_next = sched.step(eps_b, t, x_in, return_dict=False, generator=g)[0].squeeze(0)
        latents = x_next

# decode only on rank 0
videos = vae.decode(latents) if rank == 0 else None
