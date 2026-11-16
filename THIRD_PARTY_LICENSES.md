# Third-party components

Splat Repair does not ship these packages. They are installed by `npm install`,
`pip install -r core/requirements.txt` and `core/setup_deep.py`. Licences as
declared by each package at the versions used during development.

## Bundled in this repository

| Component | Licence | Where |
|---|---|---|
| Prompt (font) | SIL Open Font License 1.1 | `app/public/fonts`, `app/assets` |
| Ubuntu (font) | Ubuntu Font Licence 1.0 | `app/public/fonts` |

## App (npm)

| Package | Licence |
|---|---|
| electron | MIT |
| playcanvas | MIT |
| vite | MIT |
| esbuild | MIT |
| typescript | Apache-2.0 |
| @types/node | MIT |

Their transitive dependencies are under MIT, ISC, BSD-3-Clause, Apache-2.0 or
MPL-2.0. Electron itself bundles Chromium and Node.js under their own
licences.

## Engine (Python)

| Package | Licence | Needed for |
|---|---|---|
| numpy | BSD-3-Clause | everything |
| scipy | BSD-3-Clause | everything |
| pillow | MIT-CMU | images |
| zstandard | BSD-3-Clause | reading .spz |
| opencv-python-headless | Apache-2.0 | Deep Fill |
| torch, torchvision | BSD-3-Clause | Deep Fill |
| gsplat | Apache-2.0 | Deep Fill, rendering on the GPU |
| simple-lama-inpainting | Apache-2.0 | Deep Fill |

## Models

| Model | Licence | Note |
|---|---|---|
| LaMa (big-lama) | Apache-2.0 | Downloaded on first use by `simple-lama-inpainting`. Trained on Places2, a dataset with its own terms of use. |

## Optional external software

Fill from Photoshop drives a locally installed copy of Adobe Photoshop through
its scripting interface. Photoshop is not part of this project and needs its
own licence.
