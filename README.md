# PCB Net Segmentation

## The problem

I get a synthetic circuit board crop with a single pad marked, and I have to
segment the complete electrical net connected to that pad: the copper traces,
pads, branches and via regions that belong to it, and nothing from any other net.

What makes it hard is that the target net is not highlighted. Every net on the
board is drawn in the same style, and the seed marker only says where to start.
Other nets cross the target one and run right alongside it, so a model that
segments "copper near the seed" fails. I have to follow actual connectivity,
including where the net disappears to an internal or bottom layer and comes back.

Each sample gives me a 192x192 RGB top view with copper on a solder mask
background plus pseudo silkscreen noise, and a grayscale x-ray-like view that
sees through layers.

## What I did

The two views carry different information and the x-ray one is what makes
cross-layer tracing possible at all, so the design is about fusing them and about
treating the task as connectivity propagation from the seed rather than plain
appearance segmentation.

Every image is procedurally generated locally. No real board photos, scraped
images or copyrighted layouts are involved.

## Layout

`solution.py` is the entry point, `Approach.md` is the write up with the dataset
measurements and the literature it draws on. Datasets are not committed.
