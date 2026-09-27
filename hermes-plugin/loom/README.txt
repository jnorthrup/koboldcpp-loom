loom - Hermes context engine: /goal work as nested loops
========================================================

What it does
  Work nests like for loops. Entering a level of work pushes a frame (its stable
  instructions); leaving it pops the frame. The frames on the stack, outermost first,
  sit at the front of every request after the /goal text:

    system prompt | goal | frame 1 | frame 2 | ... | innermost frame | conversation

  Each frame is an independent obstack frame: it holds only its own text and nothing
  refers across frames. The next sibling iteration is pop + push, so only what follows
  the shared enclosing frames changes. A prefix cache on the server (koboldcpp
  --loomcache, or a provider prompt cache) keeps everything before the change. Nothing
  about frames, depth or boundaries is sent to the server.

  Compaction is the built-in compressor, unchanged; it rewrites conversation turns only.

Install
  ln -sfn "$PWD/hermes-plugin/loom" ~/.hermes/plugins/loom
  # config.yaml
  context:
    engine: loom

Use
  The model gets one tool, `loom`:
    push {text, label?}   enter a level: its stable instructions, not findings
    pop  {count?}         leave one or more levels
    show                  print the frames
  /loom in the session prints the stack.

State
  $HERMES_HOME/loom/<session_id>.json. A delegated subagent starts inside its parent's
  frames, like an inner loop.
