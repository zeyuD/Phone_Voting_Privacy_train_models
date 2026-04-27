"""
Workflow diagram with embedded figure images.

Layout:
    workflow_diagram_with_figs.py
    figures/
        mask.png
        shapes.png
        ring.png
        hue.png
        overlap.png
        final.png

Any missing image is replaced by a yellow "[ Figure: ... ]" placeholder,
so the diagram still renders while you are still collecting figures.

Dependencies:
    pip install graphviz
    Graphviz binary: https://graphviz.org/download/

Run:
    python workflow_diagram_with_figs.py
"""

import os
from graphviz import Digraph

FIG_DIR = "figures"   # change if your images live elsewhere
FIG_W, FIG_H = "1.8", "1.35"   # inches, keeps all figure nodes uniform


def fig_node(graph, node_id, filename, caption):
    """Add an image node if the file exists, else a placeholder note."""
    path = os.path.join(FIG_DIR, filename)
    if os.path.isfile(path):
        graph.node(
            node_id, "",
            shape="box",
            image=path,
            imagescale="true",
            fixedsize="true",
            width=FIG_W, height=FIG_H,
            label="",
            xlabel=caption,        # caption beside the image
            color="#4A5568",
            style="solid",
        )
    else:
        graph.node(
            node_id, f"[ Figure: {caption} ]",
            shape="note",
            fillcolor="#FFFBEA",
            color="#B7791F",
        )


g = Digraph("pipeline", format="pdf")
g.attr(rankdir="TB", splines="ortho", nodesep="0.4", ranksep="0.7")
g.attr("node", shape="box", style="rounded,filled",
       fontname="Helvetica", fontsize="11",
       fillcolor="#F5F7FA", color="#4A5568")
g.attr("edge", fontname="Helvetica", fontsize="10", color="#4A5568")

# --- Stage 1: input ---------------------------------------------------------
g.node("V",   "Video file\n(test.mp4)",           fillcolor="#E6F0FF")
g.node("F",   "Read frame\nread_frame(video, id)")
g.node("IMG", "BGR frame\n(H x W x 3)",           fillcolor="#E6F0FF")

# --- Stage 2: mask ----------------------------------------------------------
g.node("HSV",  "Convert to HSV")
g.node("THR",  "Threshold black\ninRange([0,0,0],[180,255,80])")
g.node("MORPH","Morphological CLOSE (5x5)")
g.node("MASK", "Binary black mask",               fillcolor="#FFF4E5")
fig_node(g, "FIG_MASK", "mask.png", "mask visualization")

# --- Stage 3: shape detection (parallel) -----------------------------------
with g.subgraph(name="cluster_shape") as s:
    s.attr(label="Shape detection  (findContours)", style="dashed",
           color="#718096", fontname="Helvetica-Bold", fontsize="11")
    s.node("CIR", "detect_circles()\narea>500, 20<r<150\nminEnclosingCircle")
    s.node("REC", "detect_rectangles()\narea>800, 4-vertex approx\n0.6<aspect<1.6")

g.node("CANDS", "Candidate shapes\ncircles + rects",      fillcolor="#E6F0FF")
fig_node(g, "FIG_SHAPES", "shapes.png", "raw detections")

# --- Stage 4: panel constraint ---------------------------------------------
g.node("PANEL", "is_on_panel()\nring = disk(1.4r) - disk(1.1r)\nkeep if mean_S<100 & mean_V>100")
fig_node(g, "FIG_RING", "ring.png", "ring sampling")

# --- Stage 5: color classification -----------------------------------------
g.node("COLOR", "classify_color()\nROI = 1.2r box, mean HSV\nblack/red/blue/green/orange")
fig_node(g, "FIG_HUE", "hue.png", "hue boundaries")

# --- Stage 6: overlap resolution -------------------------------------------
g.node("OVL", "remove_overlap()\ndrop circle if its center\nlies inside any rect bbox")
fig_node(g, "FIG_OVL", "overlap.png", "before / after")

# --- Stage 7: output --------------------------------------------------------
g.node("VIS", "visualize()\ndraw shape + color label")
g.node("OUT", "Annotated frame\n+ detection list",  fillcolor="#E8F5E9")
fig_node(g, "FIG_FINAL", "final.png", "final result")

# --- Edges ------------------------------------------------------------------
g.edge("V", "F")
g.edge("F", "IMG")
g.edge("IMG", "HSV")
g.edge("HSV", "THR")
g.edge("THR", "MORPH")
g.edge("MORPH", "MASK")
g.edge("MASK", "FIG_MASK", style="dotted", arrowhead="none", constraint="false")

g.edge("MASK", "CIR")
g.edge("MASK", "REC")
g.edge("CIR", "CANDS")
g.edge("REC", "CANDS")
g.edge("CANDS", "FIG_SHAPES", style="dotted", arrowhead="none", constraint="false")

g.edge("IMG", "PANEL", style="dashed", label="image\nfor HSV probe")
g.edge("CANDS", "PANEL")
g.edge("PANEL", "FIG_RING", style="dotted", arrowhead="none", constraint="false")

g.edge("PANEL", "COLOR", label="survivors")
g.edge("COLOR", "FIG_HUE", style="dotted", arrowhead="none", constraint="false")

g.edge("COLOR", "OVL")
g.edge("OVL", "FIG_OVL", style="dotted", arrowhead="none", constraint="false")

g.edge("OVL", "VIS")
g.edge("VIS", "OUT")
g.edge("OUT", "FIG_FINAL", style="dotted", arrowhead="none", constraint="false")

# Render both PDF (for LaTeX) and PNG (for slides / preview)
g.render("workflow_diagram_with_figs", cleanup=True)
g.format = "png"
g.render("workflow_diagram_with_figs", cleanup=True)
print("Wrote workflow_diagram_with_figs.pdf and .png")