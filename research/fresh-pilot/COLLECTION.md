# Collection sheet — fresh-objects-v1

Make new photographs only after this protocol and its model/code hashes have
been published. Keep the published kit unchanged. Run its packet generator
before taking any photographs. These instructions concern human collection;
software-test fixtures are not research data.

## Before starting

1. Gather two mugs, two bowls, two plates, two bottles, two spoons, two forks,
   two books and two shoes. A mug has a handle; a bowl does not. Use closed books
   and one shoe in each photograph. Choose ordinary, unmistakable examples.
2. Set aside set A for rounds 1–4 and set B for rounds 5–6. Each category must
   have two physically different objects, even if they look similar. Use the
   same A object in all four fitting rounds and the same B object in both
   development rounds. Do not swap between sets.
3. Use one camera in normal photo mode. Save original JPEG or PNG files.
   On an iPhone, select Settings → Camera → Formats → Most Compatible before capture
   ([Apple’s instructions](https://support.apple.com/en-us/116944)).
   No portrait replacement, generated fill, filters or later manual crops.
4. Generate the packet with `python3 intake.py prepare --out my-capture --collector your-alias`.
   Keep session-plan.json and CAPTURE-SHEETS.txt. The readable sheets and the plan’s `rounds` list give the order and random marker
   for each round; `caption_order` gives the later captioning order.

## Each of the six rounds

1. Record the UTC start time in rounds.csv. Copy that round's nonce from the
   session plan onto paper by hand. Photograph the paper beside all eight
   objects for that round. Save this separate original as the `marker_file`.
   It is a supporting record, never a scored image. A person must later check it.
2. Photograph the eight objects individually in the order recorded in the plan.
   Keep the target wholly inside the central square, with a little margin.
   Do not include the nonce paper, experiment labels or other target-class
   objects in these scored photographs. A normal uncluttered background is fine.
3. Use one general setup for every category in the round. Between rounds change
   the viewpoint, position, lighting or background. Do not give mugs one special
   background and books another. Avoid a burst of nearly identical photographs.
4. Keep the first usable photograph. Replace only a corrupt file, severe blur,
   the wrong object, or a target cut off by the fixed center crop, before any
   model inference. Preserve rejected originals in a separate folder and explain
   each replacement in that round's `notes`. If there are none, write `none`.
5. Put original files under the packet directory, for example in `originals/`.
   Fill `image_file` and `captured_at_utc` in samples.csv. Paths are relative to
   the packet. Preserve the provided sample IDs, classes, instances and splits.

The fixed image transform applies EXIF orientation, resizes the short side to
256 pixels, then takes the central 224 × 224 square. Keep the whole object well
away from that crop's edges. Do not run the model to select a pleasing view.

## After all 48 photographs exist

Follow `caption_order` in session-plan.json. For each photograph write a short,
factual English caption in samples.csv and record `captioned_at_utc`. Use your
own words. Describe the visible object; there is no required wording. Hide split
labels while captioning where practical. No LLM, generated template or automatic
captioning. Keep repeated descriptions and ordinary synonyms.

Use UTC timestamps with a Z suffix or +00:00 offset. Record them at the time of
each action. They must follow the actual capture and caption order. Keep the
original metadata even when you also keep a handwritten log.

## Before handing over the packet

- Count 48 scored photographs plus six separate marker photographs.
- Check every nonce and the eight objects in each marker; confirm that set B
  contains different physical objects from set A.
- Check paths, times, captions and replacement notes. Retain rejected originals.
- Include the unchanged kit and session plan. Share the packet privately; there
  is no public upload form and no need to publish personal photographs now.

The validator will reject missing slots, repeated original image bytes and
inconsistent records. It cannot check whether a marker is truthful or a caption
is human-written. State that limitation when reporting the study. New photographs
can address direct exposure to these scenes without proving independence from
the models' prior knowledge.
