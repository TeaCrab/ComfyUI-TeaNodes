import os
from ._nodes import NODE_CLASS_MAPPINGS as NCM

NODE_CLASS_MAPPINGS = {
    **NCM,
}

if os.path.exists(f"{os.path.abspath('.')}\ComfyUI\custom_nodes\ComfyUI-TeaNodes\_misc.py"):
    print('[TeaNodes]: Misc. Nodes Loaded.')
    from ._misc import NODE_CLASS_MAPPINGS as MISC_NCM
    NODE_CLASS_MAPPINGS = {
        **NCM,
        **MISC_NCM,
    }

def remove_cm_prefix(node_mapping: str) -> str:
    if node_mapping.startswith("TC_"):
        return node_mapping[3:]
    return node_mapping

NODE_DISPLAY_NAME_MAPPINGS = {key: remove_cm_prefix(key) for key in NODE_CLASS_MAPPINGS}
