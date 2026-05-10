import pickle
import numpy as np


def _selu(x):
    return 1.0507 * np.where(x > 0, x, 1.6733 * (np.exp(np.minimum(x, 50)) - 1))


def _softplus(x):
    return np.log1p(np.exp(np.clip(x, -50, 50)))


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


_ACTIVATIONS = {
    "tanh": np.tanh,
    "selu": _selu,
    "softplus": _softplus,
    "gelu": _gelu,
    "relu": lambda x: np.maximum(x, 0.0),
    "silu": lambda x: x / (1.0 + np.exp(-np.clip(x, -50, 50))),
}


def load_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _detect_vpinn_extras(state_dict):
    has_fourier = any(k == "r_embed.B" for k in state_dict)
    has_scaler = any(k.startswith("scaler_out.") for k in state_dict)
    if not (has_fourier or has_scaler):
        return None
    extras = {}
    if has_fourier:
        extras["fourier_B"] = np.asarray(state_dict["r_embed.B"])
    if has_scaler:
        extras["out_scale"] = np.asarray(state_dict["scaler_out.scale"])
        extras["out_shift"] = np.asarray(state_dict["scaler_out.shift"])
    return extras


def extract_layers(bundle):
    activation = bundle.get("activation", "tanh")

    if "torch_state_dict" in bundle and bundle["torch_state_dict"]:
        sd = {k: np.asarray(v) for k, v in bundle["torch_state_dict"].items()}
        framework = bundle.get("framework", "")
        if "deepxde" in framework:
            indices = sorted({
                int(k.split(".")[1]) for k in sd
                if "linears." in k and k.endswith(".weight")
            })
            weights = [sd[f"linears.{i}.weight"] for i in indices]
            biases = [sd[f"linears.{i}.bias"] for i in indices]
        else:
            indices = sorted({
                int(k.split(".")[1]) for k in sd
                if k.startswith("net.") and k.endswith(".weight")
            })
            weights = [sd[f"net.{i}.weight"] for i in indices]
            biases = [sd[f"net.{i}.bias"] for i in indices]
        return weights, biases, activation

    if "flax_params_numpy" in bundle:
        fp = bundle["flax_params_numpy"]
        n_dense = len(fp)
        weights = [np.asarray(fp[f"Dense_{i}"]["kernel"]).T for i in range(n_dense)]
        biases = [np.asarray(fp[f"Dense_{i}"]["bias"]) for i in range(n_dense)]
        return weights, biases, activation

    if "state_dict_np" in bundle:
        sd = bundle["state_dict_np"]
        if any("linears." in k for k in sd):
            indices = sorted({
                int(k.split(".")[1]) for k in sd
                if "linears." in k and k.endswith(".weight")
            })
            weights = [np.asarray(sd[f"linears.{i}.weight"]) for i in indices]
            biases = [np.asarray(sd[f"linears.{i}.bias"]) for i in indices]
        else:
            indices = sorted({
                int(k.split(".")[1]) for k in sd
                if k.startswith("net.") and k.endswith(".weight")
            })
            weights = [np.asarray(sd[f"net.{i}.weight"]) for i in indices]
            biases = [np.asarray(sd[f"net.{i}.bias"]) for i in indices]
        return weights, biases, activation

    if "model_state_dict" in bundle:
        sd = {k: np.asarray(v) for k, v in bundle["model_state_dict"].items()}
        extras = _detect_vpinn_extras(sd)
        indices = sorted({
            int(k.split(".")[1]) for k in sd
            if k.startswith("net.") and k.endswith(".weight")
        })
        weights = [sd[f"net.{i}.weight"] for i in indices]
        biases = [sd[f"net.{i}.bias"] for i in indices]
        if extras is not None:
            return weights, biases, activation, extras
        return weights, biases, activation

    if "params_np" in bundle:
        fp = bundle["params_np"].get("params", bundle["params_np"])
        n_dense = len(fp)
        weights = [np.asarray(fp[f"Dense_{i}"]["kernel"]).T for i in range(n_dense)]
        biases = [np.asarray(fp[f"Dense_{i}"]["bias"]) for i in range(n_dense)]
        return weights, biases, activation

    raise ValueError(f"Не распознан формат бандла. Ключи: {list(bundle.keys())[:10]}")


def forward_predict(bundle, X_input):
    layers = extract_layers(bundle)
    if len(layers) == 4:
        weights, biases, activation, extras = layers
    else:
        weights, biases, activation = layers
        extras = None

    if activation not in _ACTIVATIONS:
        raise ValueError(f"Неизвестная активация: {activation}")
    act_fn = _ACTIVATIONS[activation]

    mean_X = np.asarray(bundle["mean_X"]).reshape(-1)
    std_X = np.asarray(bundle["std_X"]).reshape(-1)
    h = (X_input.astype(np.float32) - mean_X) / std_X

    if extras is not None and "fourier_B" in extras:
        params = h[:, :5]
        r = h[:, 5:6]
        proj = r * extras["fourier_B"][None, :] * 2 * np.pi
        r_feat = np.concatenate([np.sin(proj), np.cos(proj)], axis=-1)
        h = np.concatenate([params, r_feat], axis=-1)

    for i, (W, b) in enumerate(zip(weights, biases)):
        h = h @ W.T + b
        if i < len(weights) - 1:
            h = act_fn(h)

    if extras is not None and "out_scale" in extras:
        h = h * extras["out_scale"] + extras["out_shift"]

    y_mean = np.asarray(bundle["scaler_y_mean"]).reshape(-1)
    y_std = np.asarray(bundle["scaler_y_std"]).reshape(-1)
    return h * y_std + y_mean


def predict(bundle_path, X_input):
    bundle = load_bundle(bundle_path)
    return forward_predict(bundle, X_input)
