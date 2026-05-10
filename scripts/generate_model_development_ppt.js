const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "OpenAI Codex";
pptx.company = "OpenAI";
pptx.subject = "模型开发流程";
pptx.title = "模型开发";
pptx.lang = "zh-CN";

const slide = pptx.addSlide();
slide.background = { color: "FFFFFF" };

const COLORS = {
  title: "1E334C",
  body: "1F1F1F",
  muted: "55606C",
  left: "4EA2E0",
  leftSoft: "DCEAF5",
  leftPanel: "EDF5FB",
  mid: "F69A2E",
  midSoft: "F8E7D7",
  midPanel: "FDF2E6",
  right: "92B450",
  rightSoft: "E2E8D5",
  rightPanel: "F1F5E8",
  white: "FFFFFF",
  greenText: "5B8C42",
};

const DESIGN_BOUNDS = { x: 0.5, y: 0.34, w: 12.35, h: 6.58 };
const TARGET_FRAME = { x: 2.08, y: 0.62, w: 9.17, h: 3.39 };
const SCALE = Math.min(TARGET_FRAME.w / DESIGN_BOUNDS.w, TARGET_FRAME.h / DESIGN_BOUNDS.h);
const OFFSET_X =
  TARGET_FRAME.x + (TARGET_FRAME.w - DESIGN_BOUNDS.w * SCALE) / 2 - DESIGN_BOUNDS.x * SCALE;
const OFFSET_Y =
  TARGET_FRAME.y + (TARGET_FRAME.h - DESIGN_BOUNDS.h * SCALE) / 2 - DESIGN_BOUNDS.y * SCALE;

function sx(v) {
  return OFFSET_X + v * SCALE;
}

function sy(v) {
  return OFFSET_Y + v * SCALE;
}

function ss(v) {
  return v * SCALE;
}

function addText(text, opts) {
  const scaled = { ...opts };
  if (scaled.x !== undefined) scaled.x = sx(scaled.x);
  if (scaled.y !== undefined) scaled.y = sy(scaled.y);
  if (scaled.w !== undefined) scaled.w = ss(scaled.w);
  if (scaled.h !== undefined) scaled.h = ss(scaled.h);
  if (scaled.fontSize !== undefined) scaled.fontSize = Math.max(7, scaled.fontSize * SCALE);

  slide.addText(text, {
    margin: 0,
    fontFace: "Microsoft YaHei",
    color: COLORS.body,
    ...scaled,
  });
}

function addRoundedRect(x, y, w, h, fill, extra = {}) {
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: sx(x),
    y: sy(y),
    w: ss(w),
    h: ss(h),
    rectRadius: ss(extra.rectRadius ?? 0.12),
    line:
      extra.line
        ? {
            ...extra.line,
            width: extra.line.width ? Math.max(0.5, extra.line.width * SCALE) : extra.line.width,
          }
        : { color: fill, transparency: 100 },
    fill: typeof fill === "string" ? { color: fill } : fill,
    shadow: extra.shadow
      ? {
          ...extra.shadow,
          blur: extra.shadow.blur ? Math.max(1, extra.shadow.blur * SCALE) : extra.shadow.blur,
          offset: extra.shadow.offset
            ? Math.max(0.5, extra.shadow.offset * SCALE)
            : extra.shadow.offset,
        }
      : undefined,
  });
}

function addArrow(x, y, w, h, color, transparency = 0) {
  slide.addShape(pptx.shapes.RIGHT_ARROW, {
    x: sx(x),
    y: sy(y),
    w: ss(w),
    h: ss(h),
    line: { color, transparency: 100 },
    fill: { color, transparency },
    shadow: {
      type: "outer",
      color,
      blur: Math.max(1, 2 * SCALE),
      offset: Math.max(0.5, 1 * SCALE),
      angle: 0,
      opacity: 0.15,
    },
  });
}

function addDividerLine(x, y, w) {
  slide.addShape(pptx.shapes.LINE, {
    x: sx(x),
    y: sy(y),
    w: ss(w),
    h: 0,
    line: { color: "C9D8E5", width: Math.max(0.6, 1.5 * SCALE), transparency: 10 },
  });
}

function addBadge(cx, cy, size, fillColor, text) {
  slide.addShape(pptx.shapes.OVAL, {
    x: sx(cx),
    y: sy(cy),
    w: ss(size),
    h: ss(size),
    line: { color: fillColor, transparency: 100 },
    fill: { color: fillColor, transparency: 12 },
  });
  slide.addShape(pptx.shapes.OVAL, {
    x: sx(cx + 0.07),
    y: sy(cy + 0.07),
    w: ss(size - 0.14),
    h: ss(size - 0.14),
    line: { color: fillColor, transparency: 55, width: Math.max(0.5, 1.2 * SCALE) },
    fill: { color: "FFFFFF", transparency: 100 },
  });
  addText(text, {
    x: cx,
    y: cy + 0.02,
    w: size,
    h: size - 0.02,
    align: "center",
    valign: "mid",
    color: fillColor,
    bold: true,
    fontSize: 13.5,
  });
}

function addBulletBlock(x, y, w, h, panelColor, bulletColor, lines) {
  addRoundedRect(x, y, w, h, panelColor, {
    shadow: {
      type: "outer",
      color: bulletColor,
      blur: 2,
      offset: 1,
      angle: 90,
      opacity: 0.08,
    },
  });
  lines.forEach((line, index) => {
    const yy = y + 0.2 + index * 0.34;
    slide.addShape(pptx.shapes.OVAL, {
      x: sx(x + 0.16),
      y: sy(yy + 0.08),
      w: ss(0.07),
      h: ss(0.07),
      line: { color: bulletColor, transparency: 100 },
      fill: { color: bulletColor },
    });
    addText(line, {
      x: x + 0.34,
      y: yy,
      w: w - 0.45,
      h: 0.24,
      fontSize: 16,
      color: COLORS.body,
    });
  });
}

function addSimpleItem(x, y, badgeColor, badgeText, text, width = 2.6) {
  addBadge(x, y, 0.46, badgeColor, badgeText);
  addText(text, {
    x: x + 0.64,
    y: y + 0.02,
    w: width,
    h: 0.34,
    fontSize: 16,
    color: COLORS.body,
  });
}

function addComplexItem(x, y, badgeColor, badgeText, title, detail, width) {
  addBadge(x, y, 0.62, badgeColor, badgeText);
  const lines = detail
    ? [
        {
          text: title,
          options: {
            bold: false,
            fontSize: 16,
            color: COLORS.body,
            breakLine: true,
          },
        },
        {
          text: detail,
          options: {
            fontSize: 12.5,
            color: badgeColor === COLORS.right ? COLORS.greenText : badgeColor,
            bold: false,
          },
        },
      ]
    : [{ text: title, options: { bold: false, fontSize: 16, color: COLORS.body } }];

  const textHeight = detail ? 0.8 : title.includes("\n") ? 0.82 : 0.56;

  addText(lines, {
    x: x + 0.82,
    y: y + 0.01,
    w: width,
    h: textHeight,
    margin: 0,
    fontFace: "Microsoft YaHei",
    fit: "shrink",
    breakLine: false,
    valign: "mid",
  });
}

function addCard(cfg) {
  addRoundedRect(cfg.x, cfg.y, cfg.w, cfg.h, cfg.softColor, {
    shadow: {
      type: "outer",
      color: cfg.headerColor,
      blur: 4,
      offset: 1,
      angle: 90,
      opacity: 0.08,
    },
  });

  addRoundedRect(cfg.x + 0.13, cfg.y + 0.16, cfg.w - 0.26, cfg.h - 0.32, COLORS.white, {
    shadow: {
      type: "outer",
      color: "8A8A8A",
      blur: 3,
      offset: 1,
      angle: 90,
      opacity: 0.12,
    },
  });

  addRoundedRect(cfg.x + 0.13, cfg.y + 0.16, cfg.w - 0.26, 0.62, cfg.headerColor, {
    shadow: {
      type: "outer",
      color: cfg.headerColor,
      blur: 2,
      offset: 1,
      angle: 90,
      opacity: 0.18,
    },
  });

  addText(cfg.title, {
    x: cfg.x + 0.22,
    y: cfg.y + 0.24,
    w: cfg.w - 0.44,
    h: 0.42,
    fontSize: 20,
    bold: true,
    color: "FFFFFF",
    align: "center",
    valign: "mid",
  });
}

addDividerLine(3.8, 0.82, 1.35);
addDividerLine(8.2, 0.82, 1.35);
addText("模型开发", {
  x: 4.95,
  y: 0.34,
  w: 3.45,
  h: 0.5,
  align: "center",
  valign: "mid",
  fontSize: 28,
  bold: true,
  color: COLORS.title,
});

const left = { x: 0.5, y: 1.62, w: 3.8, h: 5.3 };
const mid = { x: 4.62, y: 1.62, w: 4.2, h: 5.3 };
const right = { x: 9.15, y: 1.62, w: 3.7, h: 5.3 };

addCard({
  ...left,
  softColor: COLORS.leftSoft,
  headerColor: COLORS.left,
  title: "预训练模型基础",
});
addCard({
  ...mid,
  softColor: COLORS.midSoft,
  headerColor: COLORS.mid,
  title: "模型能力增强",
});
addCard({
  ...right,
  softColor: COLORS.rightSoft,
  headerColor: COLORS.right,
  title: "模型效率优化",
});

addBulletBlock(left.x + 0.34, left.y + 1.03, left.w - 0.68, 1.24, COLORS.leftPanel, COLORS.left, [
  "通用预训练策略：",
  "复杂场景表征",
  "大场景模拟效果更优",
]);

addSimpleItem(left.x + 0.42, left.y + 3.02, COLORS.left, "1", "通用预训练势选型");
addSimpleItem(left.x + 0.42, left.y + 3.62, COLORS.left, "2", "基线能力评测");
addSimpleItem(left.x + 0.42, left.y + 4.22, COLORS.left, "3", "目标场景适用性分析");

addComplexItem(mid.x + 0.36, mid.y + 1.04, COLORS.mid, "1", "电荷描述能力构建", "✓ 复杂场景表征提升", 2.7);
addComplexItem(mid.x + 0.36, mid.y + 2.15, COLORS.mid, "2", "长程相互作用与\n外场响应模块", "", 2.7);
addComplexItem(mid.x + 0.36, mid.y + 3.36, COLORS.mid, "3", "任务增量学习\n与体系微调", "", 2.7);

addComplexItem(right.x + 0.28, right.y + 1.04, COLORS.right, "1", "模型蒸馏与压缩", "✓ 推理效率优化", 2.15);
addComplexItem(right.x + 0.28, right.y + 2.15, COLORS.right, "2", "推理效率优化", "✓ 高性能并行", 2.15);
addComplexItem(right.x + 0.28, right.y + 3.36, COLORS.right, "3", "高性能并行实现", "", 2.15);

addArrow(3.92, 3.17, 1.05, 0.56, COLORS.left, 0);
addArrow(3.74, 3.2, 1.15, 0.5, COLORS.left, 68);
addArrow(8.6, 3.17, 0.95, 0.56, COLORS.mid, 0);
addArrow(8.46, 3.2, 1.05, 0.5, COLORS.mid, 68);
addArrow(8.6, 5.0, 0.95, 0.56, COLORS.mid, 8);
addArrow(8.46, 5.03, 1.05, 0.5, COLORS.mid, 72);

addArrow(4.2, 6.18, 2.55, 0.62, COLORS.left, 0);
addArrow(4.02, 6.2, 2.7, 0.56, COLORS.left, 74);
addArrow(6.35, 6.18, 3.35, 0.62, COLORS.mid, 0);
addArrow(6.2, 6.2, 3.55, 0.56, COLORS.mid, 74);
addArrow(9.52, 6.18, 3.05, 0.56, COLORS.right, 72);

const outputDir = path.join(__dirname, "..", "outputs");
fs.mkdirSync(outputDir, { recursive: true });

pptx.writeFile({ fileName: path.join(outputDir, "model_development_flow_compact.pptx") });
