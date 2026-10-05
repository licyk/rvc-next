"""The synthesizer and discriminator networks, ported unchanged, and the model file formats."""

import warnings

# The ported networks use torch.nn.utils.weight_norm, as the checkpoints' parameter names require.
warnings.filterwarnings("ignore", message=r"`torch\.nn\.utils\.weight_norm` is deprecated", category=FutureWarning)
