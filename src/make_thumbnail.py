"""Draw the original ChipQC writeup card (no source-dataset imagery)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/chipqc-card.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def main() -> None:
    canvas = Image.new("RGB", (560, 280), "#103f32")
    draw = ImageDraw.Draw(canvas)
    for index in range(280):
        color = (16 + index // 28, 63 + index // 24, 50 + index // 22)
        draw.line((0, index, 559, index), fill=color)
    draw.rounded_rectangle((308, 31, 528, 249), 27, fill="#e6f2e8", outline="#9acbb0", width=3)
    draw.rounded_rectangle((329, 52, 507, 228), 18, fill="#f5f8ec")
    draw.line((342, 88, 494, 88), fill="#8fb49f", width=10)
    draw.line((342, 192, 494, 192), fill="#8fb49f", width=10)
    draw.rounded_rectangle((361, 99, 475, 181), 15, fill="#d4ece0", outline="#3d8b69", width=3)
    for x, y, r in [(387, 124, 9), (426, 119, 7), (449, 144, 11),
                    (399, 153, 12), (430, 165, 6)]:
        draw.ellipse((x-r, y-r, x+r, y+r), fill="#58a77d", outline="#236b4b", width=2)
    draw.text((34, 41), "LIFE SCIENCE  /  AI4S", font=font(14, True), fill="#a7dfb9")
    draw.text((30, 99), "ChipQC", font=font(51, True), fill="#ffffff")
    draw.text((34, 163), "Organ-on-a-chip", font=font(21), fill="#d6f3df")
    draw.text((34, 193), "image review", font=font(21), fill="#d6f3df")
    draw.rounded_rectangle((34, 235, 219, 258), 11, fill="#2d7756")
    draw.text((46, 237), "DATE-HELD-OUT VALIDATION", font=font(10, True), fill="#e9ffef")
    OUT.parent.mkdir(exist_ok=True)
    canvas.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
