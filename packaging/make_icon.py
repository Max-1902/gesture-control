"""Renders packaging/AppIcon.icns: blue gradient squircle with the SF Symbol "hand.raised".

    python packaging/make_icon.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from AppKit import (NSBezierPath, NSBitmapImageRep, NSCalibratedRGBColorSpace, NSColor,
                    NSCompositingOperationSourceAtop, NSGradient, NSGraphicsContext, NSImage,
                    NSImageSymbolConfiguration, NSMakeRect, NSPNGFileType, NSRectFillUsingOperation, NSShadow)

OUT = Path(__file__).resolve().parent / "AppIcon.icns"


def render(px: int) -> NSBitmapImageRep:
    rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, px, px, 8, 4, True, False, NSCalibratedRGBColorSpace, 0, 0)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.setCurrentContext_(NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep))
    s = px / 1024.0
    inset = 100 * s   # Apple's icon grid: 824 pt artwork on a 1024 canvas
    rect = NSMakeRect(inset, inset, px - 2 * inset, px - 2 * inset)
    shape = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(rect, 185 * s, 185 * s)

    shadow = NSShadow.alloc().init()
    shadow.setShadowBlurRadius_(28 * s)
    shadow.setShadowOffset_((0, -10 * s))
    shadow.setShadowColor_(NSColor.colorWithCalibratedWhite_alpha_(0, 0.3))
    NSGraphicsContext.saveGraphicsState()
    shadow.set()
    NSColor.blackColor().set()
    shape.fill()
    NSGraphicsContext.restoreGraphicsState()

    grad = NSGradient.alloc().initWithColors_([
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.37, 0.36, 0.90, 1),
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.04, 0.52, 1.0, 1),
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.20, 0.82, 0.96, 1)])
    grad.drawInBezierPath_angle_(shape, -55)

    sym = NSImage.imageWithSystemSymbolName_accessibilityDescription_("hand.raised.fill", None)
    cfg = NSImageSymbolConfiguration.configurationWithPointSize_weight_(470 * s, 0.0)
    sym = sym.imageWithSymbolConfiguration_(cfg)
    w, h = sym.size().width, sym.size().height
    tinted = NSImage.alloc().initWithSize_((w, h))
    tinted.lockFocus()
    sym.drawInRect_(NSMakeRect(0, 0, w, h))
    NSColor.whiteColor().set()
    NSRectFillUsingOperation(NSMakeRect(0, 0, w, h), NSCompositingOperationSourceAtop)
    tinted.unlockFocus()
    tinted.drawInRect_(NSMakeRect((px - w) / 2, (px - h) / 2 - 8 * s, w, h))
    NSGraphicsContext.restoreGraphicsState()
    return rep


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "AppIcon.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                px = size * scale
                name = f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
                data = render(px).representationUsingType_properties_(NSPNGFileType, {})
                data.writeToFile_atomically_(str(iconset / name), True)
        render(1024).representationUsingType_properties_(NSPNGFileType, {}).writeToFile_atomically_(
            str(OUT.parent.parent / "docs" / "icon.png"), True)
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(OUT)], check=True)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
