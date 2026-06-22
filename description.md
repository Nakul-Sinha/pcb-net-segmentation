Overview
This is a computer vision segmentation challenge based on fully synthetic circuit board crops. Each sample shows a procedurally rendered board with copper traces, pads, vias, holes, solder mask overlays, pseudo silkscreen marks, top view imagery, an xray like view, and a seed pad marker.

The task is to segment the complete electrical net connected to the seed pad marker. The target net mask includes the connected copper traces, pads, branches, and via regions belonging to that electrical net. It does not include copper from other nets, even when traces cross visually or pass close to each other.

The target net is not highlighted in the board views. All copper nets use similar visual styling. The seed marker only identifies where the requested electrical net begins. Solvers must infer trace connectivity across top layer and internal or bottom layer evidence.

All images are created by original local procedural code. No external PCB images, board photos, scraped images, copyrighted layouts, logos, fonts, icons, hosted model images, or external visual assets were used.

Evaluation
Submissions are scored with a bounded segmentation metric:

score
=
0.45
⋅
net mask dice
+
0.25
⋅
thin trace recall
+
0.20
⋅
endpoint connectivity score
+
0.10
⋅
boundary F1
score=0.45⋅net mask dice+0.25⋅thin trace recall+0.20⋅endpoint connectivity score+0.10⋅boundary F1

For each image, net mask dice is:

dice
=
2
T
P
2
T
P
+
F
P
+
F
N
dice= 
2TP+FP+FN
2TP
​
 

Thin trace recall is recall on private centerline pixels for target trace routes:

thin trace recall
=
T
P
t
h
i
n
T
P
t
h
i
n
+
F
N
t
h
i
n
thin trace recall= 
TP 
thin
​
 +FN 
thin
​
 
TP 
thin
​
 
​
 

Endpoint connectivity score uses deterministic connected component logic. The grader finds the predicted foreground component that touches the private seed pad marker. Each private terminal pad component receives credit when at least half of its pixels are covered by that same predicted component. The mean terminal credit is multiplied by a component size factor:

size factor
=
min
⁡
(
1
,
2
⋅
A
t
a
r
g
e
t
A
c
o
m
p
o
n
e
n
t
)
size factor=min(1, 
A 
component
​
 
2⋅A 
target
​
 
​
 )

where A_{target} is the private target net mask area and A_{component} is the predicted seed connected component area. This prevents a full image foreground from receiving full connectivity credit.

Boundary F1 is the F1 score on target net boundary pixels. A predicted boundary pixel can match a true boundary pixel within a two pixel tolerance.

Direction: maximize. Range: [0, 1]. A perfect submission scores exactly 1.000000.

Dataset
Public files:

train.csv: Training rows with image_id, top_path, xray_path, seed_mask_path, and mask_path.
test.csv: Test rows with image_id, top_path, xray_path, and seed_mask_path.
sample_submission.csv: Required submission format with one row per test image.
train/images/: Training top view and xray like PNG images.
train/seed_masks/: Training binary PNG seed pad marker masks.
train/masks/: Training binary PNG target net masks.
test/images/: Test top view and xray like PNG images.
test/seed_masks/: Test binary PNG seed pad marker masks.
Column descriptions:

image_id (string): UUID like identifier for a prepared board crop.
top_path (string): Relative path to the top view circuit board image.
xray_path (string): Relative path to the xray like circuit board image.
seed_mask_path (string): Relative path to the binary seed pad marker mask.
mask_path (string): Relative path to the public training target net mask. This column appears only in train.csv.
Submission
Submit submission.csv with exactly these columns:

```text

image_id,mask_rle

68167079-a576-f6d0-0b7e-6769754c2974,1 3 193 4

Column descriptions:

image_id: The UUID like identifier from test.csv.
mask_rle: A run length encoded binary mask for the target net mask. Runs use one indexed flattened pixel positions in row major order and are written as start length pairs separated by spaces. An empty string means an empty mask.
Requirements:

Include one row for every image_id in test.csv.
Do not include duplicate image ids.
Do not include image ids outside test.csv.
Use exactly the columns image_id and mask_rle.
The decoded mask must have shape 192 by 192 pixels.
Segment the full electrical net connected to the seed marker, including connected pads, vias, branches, and traces.
Do not include non target copper traces.
Data Use And Solution Constraints
Allowed:

Public challenge files.
Public training images, seed masks, and target net masks.
Standard image processing.
Public data only computer vision and machine learning libraries.
Models trained only on the public challenge data.
Disallowed:

External PCB images, board datasets, board photos, visual assets, or layouts.
Private files, hidden routing graphs, or generator internals.
Answer key reconstruction.
Identifier, row order, file hash, visual signature, or template lookup exploits.
Modifying grade.py.
Manual reading of private answers.
External API calls at inference time.
This is a Computer Vision segmentation task. The intended solution is to infer trace connectivity from the public board views, seed marker, and released training masks.