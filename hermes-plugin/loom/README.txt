loom - Hermes context engine for planned /goal work
===================================================

What it does
  Keeps a /goal plan as a tree of prompt segments at the front of every model request.
  Segments are pushed once and never edited; each only names its parent (independent
  obstack frames, nothing interlocking). The request is laid out slowest-changing first:

    system prompt | goal + every segment, in push order | conversation | newest message + focus

  Pushing appends to the end of the trunk, so nothing already sent moves. Moving focus
  between branches only changes the newest message. A prefix cache on the server
  (koboldcpp --loomcache, or a provider prompt cache) keeps the long front and recomputes
  only the moving tip. No cache hints, tiers or segment counts are sent; the server finds
  the reusable prefix from the tokens.

  Compaction is the built-in compressor, unchanged; it rewrites conversation turns only.

Install
  ln -sfn "$PWD/hermes-plugin/loom" ~/.hermes/plugins/loom
  # config.yaml
  context:
    engine: loom

Use
  The model gets one tool, `loom`:
    plan  {nodes:[{id,text,children:[...]}], parent?}   lay down a (sub)tree, all-or-nothing
    push  {id,text}                                     child of the focus, becomes focus
    pop                                                 focus moves to the parent
    focus {id}                                          jump to any segment
    show                                                print the trunk as sent
  /loom in the session prints the tree (* = focus, | = on the focus path).

  Plan up front: a push made late in a long session re-sends the conversation after the
  trunk once. Keep findings and status in the conversation, not in segments.

State
  $HERMES_HOME/loom/<session_id>.json. Delegated subagents start from the parent's tree.
