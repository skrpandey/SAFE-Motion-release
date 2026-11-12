"""Editable draw.io sources for the manuscript diagrams (open in draw.io / diagrams.net).

Writes paper/figures/{fig_overview,fig_architecture}.drawio. Export to PDF with
  drawio -x -f pdf --crop -o paper/figures/<name>.pdf paper/figures/<name>.drawio
Style follows the SAFE-SMM figures: light boxes, one accent colour per role, short labels.
"""
import html
import os

OUT = "paper/figures"
BLUE, BLUE_F = "#0072B2", "#E6F1F8"      # data / models
ORANGE, ORANGE_F = "#D55E00", "#FDF0E6"  # evaluation design, label-tied
GREEN, GREEN_F = "#009E73", "#E5F5F0"    # quantities read out
GREY, GREY_F = "#555555", "#F5F5F5"      # neutral
FONT_BUMP = 2   # supplement diagrams are printed on landscape pages; keeps lettering near 7 pt


class Diagram:
    def __init__(self, name, w, h):
        self.name, self.w, self.h, self.cells, self.n = name, w, h, [], 2

    def _v(self, t):
        return html.escape(t).replace("\n", "&lt;br&gt;")

    def _id(self):
        self.n += 1
        return f"c{self.n}"

    def box(self, x, y, w, h, text, stroke=GREY, fill="#FFFFFF", bold=False, size=11, dashed=False,
            rounded=True, align="center"):
        i = self._id()
        style = (f"rounded={int(rounded)};whiteSpace=wrap;html=1;strokeColor={stroke};fillColor={fill};"
                 f"fontSize={size + FONT_BUMP};fontFamily=Helvetica;align={align};verticalAlign=middle;spacing=2;"
                 f"arcSize=8;{'dashed=1;' if dashed else ''}{'fontStyle=1;' if bold else ''}")
        self.cells.append(f'<mxCell id="{i}" value="{self._v(text)}" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def text(self, x, y, w, h, text, size=11, bold=False, color="#000000", align="left"):
        i = self._id()
        style = (f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;fontSize={size + FONT_BUMP};"
                 f"fontFamily=Helvetica;align={align};verticalAlign=middle;fontColor={color};"
                 f"{'fontStyle=1;' if bold else ''}")
        self.cells.append(f'<mxCell id="{i}" value="{self._v(text)}" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def group(self, x, y, w, h, label, stroke):
        i = self._id()
        style = (f"rounded=1;whiteSpace=wrap;html=1;strokeColor={stroke};fillColor=none;dashed=1;"
                 f"arcSize=3;verticalAlign=top;align=left;spacingLeft=8;spacingTop=2;fontSize={11 + FONT_BUMP};"
                 f"fontStyle=1;fontColor={stroke};fontFamily=Helvetica;")
        self.cells.insert(0, f'<mxCell id="{i}" value="{self._v(label)}" style="{style}" vertex="1" parent="1">'
                             f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def arrow(self, a, b, color=GREY, label="", dashed=False, out="r", inn="l"):
        i = self._id()
        P = {"r": (1, 0.5), "l": (0, 0.5), "t": (0.5, 0), "b": (0.5, 1), "t83": (0.83, 0), "b30": (0.3, 1), "b60": (0.6, 1)}
        (ex, ey), (nx, ny) = P[out], P[inn]
        style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;endFill=1;strokeColor={color};"
                 f"exitX={ex};exitY={ey};exitDx=0;exitDy=0;entryX={nx};entryY={ny};entryDx=0;entryDy=0;"
                 f"fontSize=9;fontFamily=Helvetica;{'dashed=1;' if dashed else ''}")
        self.cells.append(f'<mxCell id="{i}" value="{self._v(label)}" style="{style}" edge="1" parent="1" '
                          f'source="{a}" target="{b}"><mxGeometry relative="1" as="geometry"/></mxCell>')

    def save(self):
        body = "".join(self.cells)
        xml = (f'<mxfile host="drawio"><diagram name="{self.name}" id="{self.name}">'
               f'<mxGraphModel dx="{self.w}" dy="{self.h}" grid="1" gridSize="10" page="1" pageWidth="{self.w}" '
               f'pageHeight="{self.h}" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
               f'{body}</root></mxGraphModel></diagram></mxfile>')
        os.makedirs(OUT, exist_ok=True)
        path = f"{OUT}/{self.name}.drawio"
        open(path, "w").write(xml)
        print("wrote", path)


def overview():
    d = Diagram("fig_overview", 1180, 560)
    d.text(10, 4, 1160, 22, "Frozen protocols v1 to v11: every rule, margin and host fixed and committed before any "
           "model is scored (configs/protocols, REGISTRY.md)", size=11, bold=True, color=GREY, align="center")
    src = d.box(10, 60, 150, 92, "ASDPose release\n36,928 segments\n329 children\nCOCO-17 + confidence",
                BLUE, BLUE_F, bold=False)
    aud = d.box(195, 40, 175, 62, "Tracking-quality audit\n4 statistics; quality-only\nAUC 0.71 on the labels",
                ORANGE, ORANGE_F)
    qc = d.box(195, 118, 175, 62, "Frozen QC filter and\nlength protocol (windows\nof 200 frames)", ORANGE, ORANGE_F)
    d.arrow(src, aud, ORANGE); d.arrow(src, qc, ORANGE)
    # host branch
    d.group(400, 30, 480, 170, "Planted worlds on the QC host (Sections III to V)", BLUE)
    host = d.box(415, 58, 120, 62, "Host: QC-pass segments\nweak natural cue\n(0.57)", BLUE, BLUE_F)
    ops = d.box(550, 52, 150, 74, "4 operators\nmissing runs, whole joint,\nfrozen frames,\narm confidence",
                ORANGE, ORANGE_F)
    worlds = d.box(715, 52, 150, 74, "World S: tied to label\nWorld I: same marginal,\nindependent", ORANGE, ORANGE_F)
    models = d.box(415, 135, 450, 52, "CompactCTRGCN x 3 seeds per cell: none, velocity, whole-joint dropout, both, "
                   "coordinate dropout; interpolated control (v10)", BLUE, BLUE_F)
    d.arrow(qc, host, BLUE); d.arrow(host, ops); d.arrow(ops, worlds); d.arrow(worlds, models, GREY, out="b", inn="t83")
    # real branch
    d.group(400, 215, 480, 110, "Real split (Section V-E)", BLUE)
    real = d.box(415, 243, 200, 70, "Real cells, 3 seeds\n3,834 test segments, 40 children\nnatural cue present",
                 BLUE, BLUE_F)
    ref = d.box(635, 243, 230, 70, "Real-data World I reference (v10):\nsame cells retrained with independent\n"
                "missing runs on natural failures", BLUE, BLUE_F)
    d.arrow(qc, real, BLUE, out="b", inn="l"); d.arrow(real, ref, GREY, dashed=True)
    # outputs
    se = d.box(910, 40, 260, 58, "Matched-world interaction MWI = R(f_S) - R(f_I)\nmask mean, seed SD, mask range",
               GREEN, GREEN_F, bold=True)
    crossed = d.box(910, 108, 260, 44, "Crossed table: invariance is operator-specific", GREEN, GREEN_F)
    rem = d.box(910, 160, 260, 44, "Remedies: no detected reduction of MWI", GREEN, GREEN_F)
    resp = d.box(910, 250, 260, 58, "Real models: responses relative to the\nreference; reliance not identified",
                 GREEN, GREEN_F)
    d.arrow(models, se, GREEN); d.arrow(models, crossed, GREEN); d.arrow(models, rem, GREEN)
    d.arrow(ref, resp, GREEN)
    # beyond ASDPose
    d.group(10, 350, 1160, 200, "Beyond ASDPose", GREY)
    mm = d.box(25, 385, 200, 70, "MMASD (v9, v10)\nchest expansion vs squat\n11 children, OpenPose", BLUE, BLUE_F)
    utd = d.box(25, 468, 200, 70, "UTD-MHAD (v9, v10)\nwhole-body vs arm\n8 subjects, Kinect", BLUE, BLUE_F)
    bt = d.box(255, 420, 210, 84, "Boundary tests: out-of-fold\nscoring, permutation null,\nten test masks\n(pre-registered rules fail)",
               ORANGE, ORANGE_F)
    out2 = d.box(490, 420, 190, 84, "Contrast needs a stable\nreference and more\nsubjects than 8 to 11", GREEN, GREEN_F)
    ext = d.box(720, 385, 205, 70, "SSBD / ESBD external set (v11)\n127 videos, 550 windows;\nRTMPose, MediaPipe swap",
                BLUE, BLUE_F)
    tr = d.box(720, 468, 205, 70, "Transfer, confidence zeroed,\nhead refit, calibration,\nposition, label noise",
               ORANGE, ORANGE_F)
    out3 = d.box(955, 420, 210, 84, "Artifact extractor-\ndependent; transfer weak;\nheads do not transfer", GREEN, GREEN_F)
    d.arrow(mm, bt); d.arrow(utd, bt); d.arrow(bt, out2, GREEN)
    d.arrow(ext, tr, GREY, out="b", inn="t"); d.arrow(tr, out3, GREEN)
    d.save()


def architecture():
    d = Diagram("fig_architecture", 1180, 470)
    d.text(10, 4, 600, 22, "A. Where the interventions act, and the network", size=12, bold=True)
    seg = d.box(10, 40, 130, 70, "raw segment\nT x 17 x 3\n(x, y, confidence)", BLUE, BLUE_F)
    op = d.box(165, 32, 170, 86, "planted operator\n(training: World S or I;\ntest: S and I planting)\n"
               "acts before preprocessing", ORANGE, ORANGE_F)
    blind = d.box(360, 32, 150, 86, "interpolated control\nonly: interpolate\nmissing joint-frames,\ndrop confidence",
                  GREY, GREY_F, dashed=True)
    pre = d.box(535, 32, 160, 86, "200-frame window,\nimage normalisation,\nflip; channels per cell\n(table below)",
                BLUE, BLUE_F)
    d.arrow(seg, op, ORANGE); d.arrow(op, blind, GREY, dashed=True); d.arrow(blind, pre, GREY, dashed=True)
    d.text(10, 232, 580, 36, "CompactCTRGCN, 353K parameters (the published SAFE-SMM architecture), fast "
           "configuration: 25 epochs, batch 128, mixed precision", size=10, color=GREY)
    bn = d.box(10, 160, 120, 60, "input BN\n(C x V)", BLUE, BLUE_F)
    s1 = d.box(150, 160, 130, 60, "stage 1\n2 x ST block\nwidth 48", BLUE, BLUE_F)
    s2 = d.box(300, 160, 130, 60, "stage 2\n2 x ST block\nwidth 96, /2", BLUE, BLUE_F)
    s3 = d.box(450, 160, 130, 60, "stage 3\n2 x ST block\nwidth 136, /2", BLUE, BLUE_F)
    gap = d.box(600, 160, 110, 60, "global average\npool over T, V", BLUE, BLUE_F)
    fc = d.box(730, 160, 110, 60, "linear\n136 -> 2", BLUE, BLUE_F)
    out = d.box(860, 160, 120, 60, "SMM score\n(detection,\nnot diagnosis)", GREEN, GREEN_F)
    for a, b in ((bn, s1), (s1, s2), (s2, s3), (s3, gap), (gap, fc), (fc, out)):
        d.arrow(a, b, BLUE)
    d.arrow(pre, bn, BLUE, out="b", inn="t")
    feat = d.box(600, 235, 240, 44, "penultimate features: length audit,\nhead refit (DFR) on the external set",
                 GREY, GREY_F, dashed=True, size=10)
    d.arrow(gap, feat, GREY, dashed=True, out="b", inn="t")
    # cell table
    d.text(1000, 32, 170, 18, "training cells", size=11, bold=True)
    d.box(1000, 52, 170, 228,
          "none: x, y, conf (3 ch.)\n\nvelocity: + dx, dy (5 ch.)\n\nwhole-joint dropout:\njoint zeroed in window, p 0.15\n\n"
          "velocity + dropout\n(SAFE-SMM recipe)\n\ncoordinate dropout:\nx, y zeroed, conf kept\n\ninterpolated control:\nx, y (2 ch.)",
          GREY, "#FFFFFF", size=10, align="left")
    # block detail
    d.text(10, 300, 600, 22, "B. One spatial-temporal (ST) block", size=12, bold=True)
    inp = d.box(10, 360, 100, 50, "block input", GREY, GREY_F)
    g1 = d.box(140, 320, 190, 40, "CTR-GC branch 1", BLUE, BLUE_F, size=10)
    g2 = d.box(140, 368, 190, 40, "CTR-GC branch 2", BLUE, BLUE_F, size=10)
    g3 = d.box(140, 416, 190, 40, "CTR-GC branch 3", BLUE, BLUE_F, size=10)
    plus = d.box(360, 368, 40, 40, "+", GREY, "#FFFFFF", bold=True, size=14)
    bnr = d.box(425, 360, 140, 56, "BN, residual,\nReLU", GREY, GREY_F, size=10)
    t1 = d.box(595, 320, 150, 32, "5x1 conv, dilation 1", BLUE, BLUE_F, size=10)
    t2 = d.box(595, 358, 150, 32, "5x1 conv, dilation 2", BLUE, BLUE_F, size=10)
    t3 = d.box(595, 396, 150, 32, "3x1 max pool", BLUE, BLUE_F, size=10)
    t4 = d.box(595, 434, 150, 32, "1x1 conv", BLUE, BLUE_F, size=10)
    cat = d.box(775, 368, 140, 50, "concat, residual\n-> next block", GREY, GREY_F, size=10)
    for g in (g1, g2, g3):
        d.arrow(inp, g); d.arrow(g, plus)
    d.arrow(plus, bnr)
    for t in (t1, t2, t3, t4):
        d.arrow(bnr, t); d.arrow(t, cat)
    d.text(940, 340, 230, 80, "Spatial unit: three channel-wise topology-refined graph convolutions over the "
           "17 joints; temporal unit: multi-scale temporal convolution. Kinect variant: 20 joints (UTD-MHAD).",
           size=10, color=GREY)
    d.save()


def overview_design():
    """Study design without outcomes: data, operators, worlds, models, scoring and datasets.
    Orange marks only label-tied elements."""
    d = Diagram("fig_overview_design", 1180, 430)
    src = d.box(10, 40, 160, 80, "ASDPose release\n36,928 segments\n329 children\nCOCO-17 + confidence", BLUE, BLUE_F)
    qc = d.box(200, 40, 180, 80, "Published quality filter\nand length protocol\n(200-frame windows)", GREY, GREY_F)
    host = d.box(410, 20, 175, 56, "Host: quality-passing\nsegments, both classes", BLUE, BLUE_F)
    real = d.box(410, 90, 175, 56, "Real split: all segments\n(natural cue present)", BLUE, BLUE_F)
    d.arrow(src, qc); d.arrow(qc, host); d.arrow(qc, real)
    ops = d.box(615, 10, 170, 76, "Degradation operators\nmissing runs, whole joint,\nframe freezing,\narm confidence", GREY, GREY_F)
    ws = d.box(615, 100, 80, 64, "World S\nseverity tied\nto label", ORANGE, ORANGE_F)
    wi = d.box(705, 100, 80, 64, "World I\nseverity\nindependent", GREY, GREY_F)
    d.arrow(host, ops); d.arrow(ops, ws, GREY, out="b", inn="t"); d.arrow(ops, wi, GREY, out="b", inn="t")
    tr = d.box(815, 30, 170, 100, "Train the same model\nin each world\n(CompactCTRGCN,\n3 seeds, variants)", BLUE, BLUE_F)
    d.arrow(ws, tr, ORANGE, out="b", inn="b30"); d.arrow(wi, tr, GREY, out="b", inn="b60")
    sc = d.box(1005, 15, 170, 125, "Score every model\nunder label-tied (S)\nand label-independent\n(I) test degradation;\nMWI = R(f_S) - R(f_I)", GREY, GREY_F)
    d.arrow(tr, sc)
    ctl = d.box(815, 168, 360, 56, "Controls: interpolated control; operator-matched\n"
                "reference on the real split; crossed operators;\nthree test masks; subject-clustered bootstrap",
                GREY, GREY_F, size=10)
    d.arrow(real, ctl, GREY, out="b", inn="l")
    d.group(10, 230, 1160, 190, "Further datasets (boundary and transfer cases)", GREY)
    mm = d.box(25, 265, 260, 70, "MMASD: chest expansion vs squat\n11 participants, OpenPose", BLUE, BLUE_F)
    ut = d.box(25, 345, 260, 62, "UTD-MHAD: whole-body vs arm\n8 subjects, Kinect", BLUE, BLUE_F)
    bt = d.box(315, 290, 280, 90, "Same design, scored out of fold\nfor every subject, with a\npermutation null and ten test masks", GREY, GREY_F)
    ex = d.box(630, 265, 260, 70, "SSBD / ESBD external set\n127 videos, 550 windows", BLUE, BLUE_F)
    pp = d.box(630, 345, 260, 62, "Two extraction pipelines:\nRTMPose with child selection; MediaPipe", GREY, GREY_F)
    tt = d.box(920, 285, 235, 100, "Transfer of the ASDPose models,\nconfidence channel zeroed,\nhead refit, calibration,\nposition and label audits", GREY, GREY_F)
    d.arrow(mm, bt); d.arrow(ut, bt); d.arrow(ex, pp, GREY, out="b", inn="t"); d.arrow(pp, tt)
    d.save()


if __name__ == "__main__":
    overview()
    architecture()
    overview_design()
