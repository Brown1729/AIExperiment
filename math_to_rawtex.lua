--[[
  math_to_rawtex.lua — Prevent pandoc from double-wrapping LaTeX math
  environments that are already self-contained display math.

  Problem:  $$ \begin{align*} ... \end{align*} $$
  Pandoc wraps it in \[...\] producing:
      \[\begin{align*} ... \end{align*}\]
  which is invalid LaTeX — amsmath nesting error.

  These top-level environments must be emitted raw, without any wrapper.
  Inner environments (aligned, split, cases, matrix, etc.) still need the
  \[...\] or $$ wrapper and are left alone.
]]

-- Self-contained display math environments that must NOT be wrapped.
local top_level_envs = {
  "align*",
  "align",
  "alignat",
  "alignat*",
  "equation*",
  "equation",
  "flalign*",
  "flalign",
  "gather*",
  "gather",
  "multline*",
  "multline",
}

local prefixes = {}
for _, env in ipairs(top_level_envs) do
  prefixes[#prefixes + 1] = "\\begin{" .. env .. "}"
end

local function contains_top_level_env(text)
  for _, prefix in ipairs(prefixes) do
    if text:find(prefix, 1, true) then
      return true
    end
  end
  return false
end

-- Block-level: standalone $$...$$ paragraph → emit the math text directly
-- (no $$ wrapper — the environment is self-contained).
function Para(el)
  if #el.content == 1 and el.content[1].t == "Math" then
    local m = el.content[1]
    if m.mathtype == "DisplayMath" and contains_top_level_env(m.text) then
      return pandoc.RawBlock("latex", m.text)
    end
  end
end

-- Inline-level fallback: display math embedded in a mixed-content paragraph.
function Math(el)
  if el.mathtype ~= "DisplayMath" then
    return nil
  end
  if contains_top_level_env(el.text) then
    return pandoc.RawInline("latex", el.text)
  end
  return nil
end
