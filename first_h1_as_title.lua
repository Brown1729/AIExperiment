local function contains_cjk(text)
  for _, codepoint in utf8.codes(text) do
    if (codepoint >= 0x4E00 and codepoint <= 0x9FFF)
      or (codepoint >= 0x3400 and codepoint <= 0x4DBF)
      or (codepoint >= 0xF900 and codepoint <= 0xFAFF) then
      return true
    end
  end
  return false
end

local function normalize_text_math(text)
  local normalized = text
  normalized = normalized:gsub("\\rightarrow", " -> ")
  normalized = normalized:gsub("\\to", " -> ")
  normalized = normalized:gsub("%s+", " ")
  normalized = normalized:gsub("^%s+", "")
  normalized = normalized:gsub("%s+$", "")
  return normalized
end

local function meta_truthy(value)
  if value == nil then
    return false
  end

  if type(value) == "boolean" then
    return value
  end

  local kind = value.t
  if kind == "MetaBool" then
    return value
  end

  local text = pandoc.utils.stringify(value):lower()
  return text == "true" or text == "1" or text == "yes" or text == "on"
end

local function strip_anchor_links(inlines)
  local cleaned = {}
  local pilcrow = utf8.char(0x00B6)

  for _, inline in ipairs(inlines) do
    if inline.t == "Link" then
      local classes = {}
      local link_text = pandoc.utils.stringify(inline)
      local is_anchor_link = false

      if inline.attr ~= nil and inline.attr.classes ~= nil then
        classes = inline.attr.classes
      elseif inline.c ~= nil and inline.c[1] ~= nil and inline.c[1][2] ~= nil then
        classes = inline.c[1][2]
      end

      for _, class_name in ipairs(classes) do
        if class_name == "anchor-link" then
          is_anchor_link = true
          break
        end
      end

      if not is_anchor_link and link_text == pilcrow then
        is_anchor_link = true
      end

      if not is_anchor_link then
        table.insert(cleaned, inline)
      end
    else
      table.insert(cleaned, inline)
    end
  end

  while #cleaned > 0 and cleaned[#cleaned].t == "Space" do
    table.remove(cleaned)
  end

  return cleaned
end

local function is_break_inline(inline)
  return inline.t == "SoftBreak" or inline.t == "LineBreak"
end

local function is_spacing_inline(inline)
  return inline.t == "Space" or is_break_inline(inline)
end

local function trim_inlines(inlines)
  local start_index = 1
  local end_index = #inlines

  while start_index <= end_index and is_spacing_inline(inlines[start_index]) do
    start_index = start_index + 1
  end

  while end_index >= start_index and is_spacing_inline(inlines[end_index]) do
    end_index = end_index - 1
  end

  local trimmed = {}

  for index = start_index, end_index do
    table.insert(trimmed, inlines[index])
  end

  return trimmed
end

local function is_image_only_line(inlines)
  local trimmed = trim_inlines(inlines)
  return #trimmed == 1 and trimmed[1].t == "Image"
end

local function normalize_block_spacing(block)
  local trimmed = trim_inlines(block.content)

  if #trimmed == #block.content then
    return nil
  end

  block.content = trimmed
  return block
end

local function split_image_only_lines(block)
  local lines = {}
  local current_line = {}

  for _, inline in ipairs(block.content) do
    if is_break_inline(inline) then
      table.insert(lines, current_line)
      current_line = {}
    else
      table.insert(current_line, inline)
    end
  end

  table.insert(lines, current_line)

  local has_image_only_line = false

  for _, line in ipairs(lines) do
    if is_image_only_line(line) then
      has_image_only_line = true
      break
    end
  end

  if not has_image_only_line then
    return nil
  end

  local constructor = block.t == "Plain" and pandoc.Plain or pandoc.Para
  local blocks = {}
  local pending = {}

  local function flush_pending()
    local trimmed = trim_inlines(pending)
    if #trimmed > 0 then
      table.insert(blocks, constructor(trimmed))
    end
    pending = {}
  end

  local function append_text_line(line)
    local trimmed = trim_inlines(line)
    if #trimmed == 0 then
      return
    end

    if #pending > 0 then
      table.insert(pending, pandoc.SoftBreak())
    end

    for _, inline in ipairs(trimmed) do
      table.insert(pending, inline)
    end
  end

  for _, line in ipairs(lines) do
    if is_image_only_line(line) then
      flush_pending()
      table.insert(blocks, constructor(trim_inlines(line)))
    else
      append_text_line(line)
    end
  end

  flush_pending()

  return blocks
end

function Pandoc(doc)
  local force_title_from_h1 = meta_truthy(doc.meta["title-from-first-h1"])

  if doc.meta.title and not force_title_from_h1 then
    return doc
  end

  local captured_title = nil
  local saw_title_header = false

  doc = doc:walk({
    Header = function(el)
      if not saw_title_header and el.level == 1 then
        captured_title = pandoc.MetaInlines(strip_anchor_links(el.content))
        saw_title_header = true
        return {}
      end

      if saw_title_header then
        el.level = math.max(1, el.level - 1)
      end

      el.content = strip_anchor_links(el.content)

      return el
    end
  })

  doc = doc:walk({
    Para = normalize_block_spacing,
    Plain = normalize_block_spacing,
  })

  doc = doc:walk({
    Para = split_image_only_lines,
    Plain = split_image_only_lines,
  })

  if captured_title ~= nil then
    doc.meta.title = captured_title
  end

  return doc
end

function Math(el)
  if contains_cjk(el.text) then
    return pandoc.Str(normalize_text_math(el.text))
  end
  return el
end
