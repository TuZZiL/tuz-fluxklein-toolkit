import json

workflow = {
    "last_node_id": 24,
    "last_link_id": 30,
    "nodes": [
        {
            "id": 1, "type": "LoadImage",
            "pos": [50, 300],
            "size": [300, 100],
            "widgets_values": ["example.png", "image"],
            "outputs": [
                {"name": "IMAGE", "type": "IMAGE", "links": [1, 2], "slot_index": 0},
                {"name": "MASK", "type": "MASK", "links": None}
            ]
        },
        {
            "id": 2, "type": "UnetLoaderGGUF",
            "pos": [50, 50],
            "size": [350, 60],
            "widgets_values": ["FLUX/flux-2-klein-9b-Q8_0.gguf"],
            "outputs": [
                {"name": "MODEL", "type": "MODEL", "links": [3], "slot_index": 0}
            ]
        },
        {
            "id": 3, "type": "CLIPLoader",
            "pos": [50, 500],
            "size": [350, 80],
            "widgets_values": ["qwen_3_8b_fp8mixed.safetensors", "flux2", "default"],
            "outputs": [
                {"name": "CLIP", "type": "CLIP", "links": [4], "slot_index": 0}
            ]
        },
        {
            "id": 4, "type": "VAELoader",
            "pos": [50, 650],
            "size": [300, 60],
            "widgets_values": ["flux2-vae.safetensors"],
            "outputs": [
                {"name": "VAE", "type": "VAE", "links": [5, 6], "slot_index": 0}
            ]
        },
        {
            "id": 5, "type": "FluxLoraScheduled",
            "pos": [450, 50],
            "size": [350, 240],
            "widgets_values": [
                "FLUX9bKlein/removedress6000_6.safetensors",
                0.8,
                "Edit",
                "Fade Out",
                "Auto",
                0.5,
                True,
                5
            ],
            "inputs": [
                {"name": "model", "type": "MODEL", "link": 3},
                {"name": "conditioning", "type": "CONDITIONING", "link": 9}
            ],
            "outputs": [
                {"name": "model", "type": "MODEL", "links": [7], "slot_index": 0},
                {"name": "conditioning", "type": "CONDITIONING", "links": [14, 24], "slot_index": 1}
            ]
        },
        {
            "id": 6, "type": "CLIPTextEncode",
            "pos": [450, 500],
            "size": [350, 100],
            "widgets_values": ["remove dress from woman"],
            "inputs": [
                {"name": "clip", "type": "CLIP", "link": 4}
            ],
            "outputs": [
                {"name": "CONDITIONING", "type": "CONDITIONING", "links": [9], "slot_index": 0}
            ]
        },
        {
            "id": 8, "type": "ConditioningZeroOut",
            "pos": [850, 560],
            "size": [300, 50],
            "inputs": [
                {"name": "conditioning", "type": "CONDITIONING", "link": 24}
            ],
            "outputs": [
                {"name": "CONDITIONING", "type": "CONDITIONING", "links": [28], "slot_index": 0}
            ]
        },
        {
            "id": 9, "type": "ImageScaleToTotalPixels",
            "pos": [450, 350],
            "size": [300, 80],
            "widgets_values": ["nearest-exact", 1.0, 1],
            "inputs": [
                {"name": "image", "type": "IMAGE", "link": 1}
            ],
            "outputs": [
                {"name": "IMAGE", "type": "IMAGE", "links": [11], "slot_index": 0}
            ]
        },
        {
            "id": 10, "type": "VAEEncode",
            "pos": [850, 300],
            "size": [250, 50],
            "inputs": [
                {"name": "pixels", "type": "IMAGE", "link": 11},
                {"name": "vae", "type": "VAE", "link": 5}
            ],
            "outputs": [
                {"name": "LATENT", "type": "LATENT", "links": [12, 13], "slot_index": 0}
            ]
        },
        {
            "id": 11, "type": "ReferenceLatent",
            "pos": [1200, 400],
            "size": [250, 80],
            "inputs": [
                {"name": "conditioning", "type": "CONDITIONING", "link": 14},
                {"name": "latent", "type": "LATENT", "link": 12}
            ],
            "outputs": [
                {"name": "CONDITIONING", "type": "CONDITIONING", "links": [15], "slot_index": 0}
            ]
        },
        {
            "id": 12, "type": "ReferenceLatent",
            "pos": [1200, 520],
            "size": [250, 80],
            "inputs": [
                {"name": "conditioning", "type": "CONDITIONING", "link": 28},
                {"name": "latent", "type": "LATENT", "link": 13}
            ],
            "outputs": [
                {"name": "CONDITIONING", "type": "CONDITIONING", "links": [16], "slot_index": 0}
            ]
        },
        {
            "id": 13, "type": "GetImageSize",
            "pos": [450, 180],
            "size": [200, 50],
            "inputs": [
                {"name": "image", "type": "IMAGE", "link": 2}
            ],
            "outputs": [
                {"name": "width", "type": "INT", "links": [16, 18], "slot_index": 0},
                {"name": "height", "type": "INT", "links": [17, 19], "slot_index": 1}
            ]
        },
        {
            "id": 14, "type": "Flux2Scheduler",
            "pos": [1200, 200],
            "size": [250, 100],
            "widgets_values": [4, 1024, 1024],
            "inputs": [
                {"name": "width", "type": "INT", "link": 16},
                {"name": "height", "type": "INT", "link": 17}
            ],
            "outputs": [
                {"name": "SIGMAS", "type": "SIGMAS", "links": [20], "slot_index": 0}
            ]
        },
        {
            "id": 15, "type": "EmptyFlux2LatentImage",
            "pos": [1200, 100],
            "size": [250, 80],
            "widgets_values": [1024, 1024, 1],
            "inputs": [
                {"name": "width", "type": "INT", "link": 18},
                {"name": "height", "type": "INT", "link": 19}
            ],
            "outputs": [
                {"name": "LATENT", "type": "LATENT", "links": [21], "slot_index": 0}
            ]
        },
        {
            "id": 16, "type": "CFGGuider",
            "pos": [1500, 300],
            "size": [250, 120],
            "widgets_values": [1],
            "inputs": [
                {"name": "model", "type": "MODEL", "link": 7},
                {"name": "positive", "type": "CONDITIONING", "link": 15},
                {"name": "negative", "type": "CONDITIONING", "link": 16}
            ],
            "outputs": [
                {"name": "GUIDER", "type": "GUIDER", "links": [22], "slot_index": 0}
            ]
        },
        {
            "id": 17, "type": "KSamplerSelect",
            "pos": [1500, 50],
            "size": [250, 60],
            "widgets_values": ["lcm"],
            "outputs": [
                {"name": "SAMPLER", "type": "SAMPLER", "links": [23], "slot_index": 0}
            ]
        },
        {
            "id": 18, "type": "RandomNoise",
            "pos": [1500, 150],
            "size": [250, 80],
            "widgets_values": [42, "randomize"],
            "outputs": [
                {"name": "NOISE", "type": "NOISE", "links": [25], "slot_index": 0}
            ]
        },
        {
            "id": 19, "type": "SamplerCustomAdvanced",
            "pos": [1800, 200],
            "size": [300, 150],
            "inputs": [
                {"name": "noise", "type": "NOISE", "link": 25},
                {"name": "guider", "type": "GUIDER", "link": 22},
                {"name": "sampler", "type": "SAMPLER", "link": 23},
                {"name": "sigmas", "type": "SIGMAS", "link": 20},
                {"name": "latent_image", "type": "LATENT", "link": 21}
            ],
            "outputs": [
                {"name": "output", "type": "LATENT", "links": [26], "slot_index": 0},
                {"name": "denoised_output", "type": "LATENT", "links": None}
            ]
        },
        {
            "id": 20, "type": "VAEDecode",
            "pos": [2150, 200],
            "size": [250, 50],
            "inputs": [
                {"name": "samples", "type": "LATENT", "link": 26},
                {"name": "vae", "type": "VAE", "link": 6}
            ],
            "outputs": [
                {"name": "IMAGE", "type": "IMAGE", "links": [27], "slot_index": 0}
            ]
        },
        {
            "id": 21, "type": "SaveImage",
            "pos": [2450, 200],
            "size": [300, 300],
            "widgets_values": ["Flux2-Klein-Scheduled"],
            "inputs": [
                {"name": "images", "type": "IMAGE", "link": 27}
            ]
        }
    ],
    "links": [
        [1, 1, 0, 9, 0, "IMAGE"],
        [2, 1, 0, 13, 0, "IMAGE"],
        [3, 2, 0, 5, 0, "MODEL"],
        [4, 3, 0, 6, 0, "CLIP"],
        [5, 4, 0, 10, 1, "VAE"],
        [6, 4, 0, 20, 1, "VAE"],
        [7, 5, 0, 16, 0, "MODEL"],
        [9, 6, 0, 5, 1, "CONDITIONING"],
        [11, 9, 0, 10, 0, "IMAGE"],
        [12, 10, 0, 11, 1, "LATENT"],
        [13, 10, 0, 12, 1, "LATENT"],
        [14, 5, 1, 11, 0, "CONDITIONING"],
        [15, 11, 0, 16, 1, "CONDITIONING"],
        [16, 12, 0, 16, 2, "CONDITIONING"],
        [16, 13, 0, 14, 0, "INT"],
        [17, 13, 1, 14, 1, "INT"],
        [18, 13, 0, 15, 0, "INT"],
        [19, 13, 1, 15, 1, "INT"],
        [20, 14, 0, 19, 3, "SIGMAS"],
        [21, 15, 0, 19, 4, "LATENT"],
        [22, 16, 0, 19, 1, "GUIDER"],
        [23, 17, 0, 19, 2, "SAMPLER"],
        [24, 5, 1, 8, 0, "CONDITIONING"],
        [25, 18, 0, 19, 0, "NOISE"],
        [26, 19, 0, 20, 0, "LATENT"],
        [27, 20, 0, 21, 0, "IMAGE"],
        [28, 8, 0, 12, 0, "CONDITIONING"]
    ],
    "groups": [],
    "config": {},
    "extra": {},
    "version": 0.4
}

with open("workflow_scheduled_lora.json", "w") as f:
    json.dump(workflow, f, indent=2)
print(f"Saved: workflow_scheduled_lora.json ({len(workflow['nodes'])} nodes, {len(workflow['links'])} links)")
