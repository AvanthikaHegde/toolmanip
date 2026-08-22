   A zero-shot robot tool manipulation pipeline for industrial settings, built on
  vision-language models. Given a photo of a workbench and a board of available
  tools, the system decides what the task requires, picks the right tool, and works
  out exactly where to grip it and how to move it — without task-specific training
  data. It's an implementation of Zhou et al., *"Towards zero-shot robot tool
  manipulation in industrial context"*, Robotics and Computer-Integrated
  Manufacturing 98 (2026).

  The pipeline runs in two modules. **Module I (Task Planning)** prompts a VLM
  through four stages — task understanding, tool identification, tool selection,
  and step planning — producing a structured plan with the target object, the
  chosen tool, and per-step success criteria.
  
  **Module II (Affordance Reasoning)**
  then locates the keypoints that make the plan executable: a grasp point `H` and
  functional point `F` on the tool, a start point `O` and end point `Q` on the
  target, plus direction angles and an operation vector. It does this via
  Hierarchical Region Extraction (HRE), which uses LangSAM to segment regions of
  interest, and a Structured Interaction Field (SIF), which annotates candidate
  keypoints onto those crops as a visual prompt for the VLM. Coordinates come out
  in both pixel and normalized form, ready for pinhole projection to 3D once a
  depth camera is attached.

  ## Setup

  ```bash
  pip install -r requirements.txt
  pip install git+https://github.com/luca-medeiros/lang-segment-anything.git  # optional, for HRE

  Set OPENAI_API_KEY in a .env file, then run python main.py. HRE degrades
  gracefully if LangSAM isn't installed — it falls back to using the full image as
  the ROI with a warning rather than crashing. To try the pipeline without an API
  key, set IMAGE_MODE=mock in .env. Results are written to results/ and run
  history to memory/, both git-ignored.



Reference -https://github.com/toolmanip/ToolManip
