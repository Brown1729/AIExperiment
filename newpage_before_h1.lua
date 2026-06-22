-- Insert \newpage before the first H1 heading so the main content
-- starts on a fresh page (after the TOC).

local done = false

function Header(el)
  if not done and el.level == 1 then
    done = true
    return {
      pandoc.RawBlock("latex", "\\newpage"),
      el,
    }
  end
end
