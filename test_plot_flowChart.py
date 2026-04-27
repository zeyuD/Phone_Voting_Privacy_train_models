"""
Workflow diagram generator for the black-panel shape detection pipeline.

Dependencies:
    pip install graphviz
    (and the Graphviz binary: https://graphviz.org/download/)

Run:
    python workflow_diagram.py
    -> produces workflow_diagram.pdf  and  workflow_diagram.png
"""

from graphviz import Digraph

g = Digraph("pipeline", format="pdf")
g.attr(rankdir="TB", splines="ortho", nodesep="0.35", ranksep="0.55")
g.attr("node", shape="box", style="rounded,filled",
       fontname="Helvetica", fontsize="11",
       fillcolor="#F5F7FA", color="#4A5568")
g.attr("edge", fontname="Helvetica", fontsize="10", color="#4A5568")

# --- Stage 1: input ---------------------------------------------------------
g.node("V",   "Video file\n(test.mp4)",           fillcolor="#E6F0FF")
g.node("F",   "Read frame\nread_frame(video, id)", shape="box")
g.node("IMG", "BGR frame\n(H x W x 3)",           fillcolor="#E6F0FF")

# --- Stage 2: mask ----------------------------------------------------------
g.node("HSV",  "Convert to HSV")
g.node("THR",  "Threshold black\ninRange([0,0,0],[180,255,80])")
g.node("MORPH","Morphological CLOSE (5x5)")
g.node("MASK", "Binary black mask",               fillcolor="#FFF4E5")
# Placeholder note for a figure
g.node("FIG_MASK", "[ Figure: mask visualization ]",
       shape="note", fillcolor="#FFFBEA", color="#B7791F")

# --- Stage 3: shape detection (parallel) -----------------------------------
with g.subgraph(name="cluster_shape") as s:
    s.attr(label="Shape detection  (findContours)", style="dashed",
           color="#718096", fontname="Helvetica-Bold", fontsize="11")
    s.node("CIR", "detect_circles()\narea>500, 20<r<150\nminEnclosingCircle")
    s.node("REC", "detect_rectangles()\narea>800, 4-vertex approx\n0.6<aspect<1.6")

g.node("CANDS", "Candidate shapes\ncircles + rects",      fillcolor="#E6F0FF")
g.node("FIG_SHAPES", "[ Figure: raw detections ]",
       shape="note", fillcolor="#FFFBEA", color="#B7791F")

# --- Stage 4: panel constraint ---------------------------------------------
g.node("PANEL", "is_on_panel()\nring = disk(1.4r) - disk(1.1r)\nkeep if mean_S<100 & mean_V>100")
g.node("FIG_RING", "[ Figure: ring sampling ]",
       shape="note", fillcolor="#FFFBEA", color="#B7791F")

# --- Stage 5: color classification -----------------------------------------
g.node("COLOR", "classify_color()\nROI = 1.2r box, mean HSV\nblack / red / blue / green / orange")
g.node("FIG_HUE", "[ Figure: hue boundaries ]",
       shape="note", fillcolor="#FFFBEA", color="#B7791F")

# --- Stage 6: overlap resolution -------------------------------------------
g.node("OVL", "remove_overlap()\ndrop circle if its center\nlies inside any rect bbox")
g.node("FIG_OVL", "[ Figure: before / after ]",
       shape="note", fillcolor="#FFFBEA", color="#B7791F")

# --- Stage 7: output --------------------------------------------------------
g.node("VIS", "visualize()\ndraw shape + color label")
g.node("OUT", "Annotated frame\n+ detection list",  fillcolor="#E8F5E9")
g.node("FIG_FINAL", "[ Figure: final result ]",
       shape="note", fillcolor="#FFFBEA", color="#B7791F")

# --- Edges ------------------------------------------------------------------
g.edge("V", "F")
g.edge("F", "IMG")
g.edge("IMG", "HSV")
g.edge("HSV", "THR")
g.edge("THR", "MORPH")
g.edge("MORPH", "MASK")
g.edge("MASK", "FIG_MASK", style="dotted", arrowhead="none")

g.edge("MASK", "CIR")
g.edge("MASK", "REC")
g.edge("CIR", "CANDS")
g.edge("REC", "CANDS")
g.edge("CANDS", "FIG_SHAPES", style="dotted", arrowhead="none")

g.edge("IMG", "PANEL", style="dashed", label="image\nfor HSV probe")
g.edge("CANDS", "PANEL")
g.edge("PANEL", "FIG_RING", style="dotted", arrowhead="none")

g.edge("PANEL", "COLOR", label="survivors")
g.edge("COLOR", "FIG_HUE", style="dotted", arrowhead="none")

g.edge("COLOR", "OVL")
g.edge("OVL", "FIG_OVL", style="dotted", arrowhead="none")

g.edge("OVL", "VIS")
g.edge("VIS", "OUT")
g.edge("OUT", "FIG_FINAL", style="dotted", arrowhead="none")

# Render
g.render("workflow_diagram", cleanup=True)        # PDF
g.format = "png"
g.render("workflow_diagram", cleanup=True)        # PNG
print("Wrote workflow_diagram.pdf and workflow_diagram.png")