import omni.replicator.core as rep
import omni.usd
import time
import os

# ─── ONLY CHANGE THIS EACH SESSION ───────────────────────────────────────────
BATCH_INDEX = 8        # 0-5 positive (disk visible), 6-8 negative (disk hidden)
# ─────────────────────────────────────────────────────────────────────────────

FRAMES_PER_BATCH = 1000
OUTPUT_ROOT      = "C:/landing_pad_3"

batch_dir = os.path.join(OUTPUT_ROOT, f"batch_{BATCH_INDEX:04d}")
os.makedirs(batch_dir, exist_ok=True)
print(f"Output folder: {batch_dir}")

# Enable motion blur
stage = omni.usd.get_context().get_stage()
render_settings = stage.GetPrimAtPath("/Render/OmniverseGlobalRenderSettings")
render_settings.GetAttribute("disableMotionBlur").Set(False)
render_settings.GetAttribute("instantaneousShutter").Set(False)
print("Motion blur enabled")

with rep.new_layer():
    camera = rep.create.camera(
        position=(0, 0, 700),
        look_at=(0, 0, 253)
    )
    render_product = rep.create.render_product(camera, resolution=(640, 480))

    landing_pad = rep.get.prims("/World/Disk")
    with landing_pad:
        rep.modify.semantics([("class", "landing_pad")])

    sun_light  = rep.get.prims("/World/SunLight")
    dome_light = rep.get.prims("/World/DomeLight")

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=batch_dir,
        rgb=True,
        bounding_box_2d_tight=True,
    )
    writer.attach([render_product])

    with rep.trigger.on_frame(num_frames=FRAMES_PER_BATCH, rt_subframes=4):
        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform(
                    (-400, -400, 300),
                    ( 400,  400, 2700),
                ),
                look_at=rep.distribution.uniform(
                    (-60, -60, 253),
                    ( 60,  60, 253),
                )
            )
        with sun_light:
            rep.modify.attribute(
                "inputs:intensity",
                rep.distribution.uniform(1000.0, 15000.0)
            )
            rep.modify.attribute(
                "inputs:color",
                rep.distribution.uniform(
                    (1.0, 0.8, 0.6),
                    (1.0, 1.0, 1.0),
                )
            )
            rep.modify.pose(
                rotation=rep.distribution.uniform(
                    (-85, 0, 0),
                    (-5, 360, 0)
                )
            )
        with dome_light:
            rep.modify.attribute(
                "inputs:intensity",
                rep.distribution.uniform(250.0, 500.0)
            )
            rep.modify.attribute(
                "inputs:color",
                rep.distribution.uniform(
                    (0.7, 0.8, 1.0),
                    (1.0, 1.0, 1.0),
                )
            )

rep.orchestrator.preview()
time.sleep(15)
rep.orchestrator.run()
time.sleep(120)
print(f"✓ Batch {BATCH_INDEX:04d} done — {FRAMES_PER_BATCH} frames saved to {batch_dir}")
print(f"  Next: set BATCH_INDEX = {BATCH_INDEX + 1} and run in fresh session")
