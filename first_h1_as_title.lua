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
