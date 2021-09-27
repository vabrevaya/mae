--
--  Helper functions for handling strings
-- 

local str = {}

-------------------------------------------------------------------
--  Split string according to the separator
--
--  Input: 
--    inputstr  : string to split
--    sep       : character by which to split. If not specified then
--                the white space (%s) will be used as separator.
--
--  Returns:
--    parts     : table with the splitted string
--                (only one element if the separator is not present)
--
--  Raises error if inputstr is nil
-------------------------------------------------------------------
function str.split(inputstr, sep)
    assert(inputstr ~= nil, "No string provided")

    -- default separator
    if sep == nil then
         sep = "%s"
    end

    -- split and save in table
    local parts = {} ; i = 1

    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
         parts[i] = str
         i = i + 1
    end

    return parts
end

-------------------------------------------------------------------
--  Whitespace strip
-------------------------------------------------------------------
function str.strip(str)
    return str:gsub("%s+", "")
end

return str