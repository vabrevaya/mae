--
--  Helper functions for handling file IO.
--
--  Some were taken from http://lua-users.org/wiki/FileInputOutput
--

local M = {}

-------------------------------------------------------
-- Find the length of a file
--   filename: file name
-- returns
--   len: length of file
--   asserts on error
-------------------------------------------------------
function M.length_of_file(filename)
  local fh = assert(io.open(filename, "rb"))
  local len = assert(fh:seek("end"))
  fh:close()
  return len
end


-------------------------------------------------------
-- Return true if file exists and is readable.
-------------------------------------------------------
function M.file_exists(path)
  local file = io.open(path, "rb")
  if file then file:close() end
  return file ~= nil
end


-------------------------------------------------------
-- Guarded seek.
-- Same as file:seek except throws string
-- on error.
-- Requires Lua 5.1.
-------------------------------------------------------
function M.seek(fh, ...)
  assert(fh:seek(...))
end


-------------------------------------------------------
-- get all lines from a file, returns an empty 
-- list/table if the file does not exist
-------------------------------------------------------
function M.read_lines(file)
  if not M.file_exists(file) then return {} end
  
  lines = {}

  for line in io.lines(file) do 
    lines[#lines + 1] = line
  end

  return lines
  
end


-------------------------------------------------------
-- Returns the base file name from a full path,
-- without its extension
-------------------------------------------------------
function M.get_filename_wo_extension(filepath)

  local filename = paths.basename(filepath)
  local file_wo_ext, _ = M.remove_extension(filename)
  return file_wo_ext
end


-------------------------------------------------------
-- Remove extension from a file name
-- Returns the filename without extension, 
-- and the extension
-------------------------------------------------------
function M.remove_extension(filename)

  local ext = paths.extname(filename)

  if ext == nil then
    return filename
  else
    local file_wo_ext = filename:sub(1, -(#ext + 2))
    return file_wo_ext, ext
  end

end


-------------------------------------------------------
-- Copy file from  source to target
-------------------------------------------------------
function M.cp(filepath_from, filepath_to)

  -- note: probably not the most efficient way...
  local cp_cmd = 'cp ' .. filepath_from .. ' ' .. filepath_to
  sys.execute(cp_cmd)

end


-------------------------------------------------------
-- Read file containing a list of integers, 
-- one per line. 
-- Returns a table with the list of indices.
--
-- Set 'to_1_based' to True if indices are 0-based but the
-- output must be 1-based.
-- 
-- 'output_type' : 'set' | 'list'
--
-------------------------------------------------------

function M.load_index_file(filepath, to_1_based, output_type)
  output_type = output_type or 'list'
  assert(output_type == 'set' or output_type == 'list', "Invalid output type: " .. output_type)

  local res = {}
  local as_set = output_type == 'set'

  local f = assert(io.open(filepath, 'r'), "Unable to open file for reading: " .. filepath)

  while true do
    local line = f:read()
    if line == nil then break end

    local ind = tonumber(line)
    if ind == nil then
      error("Invalid line encountered in index file (unable to convert to number): " .. line)
    end

    if to_1_based then
      ind = ind+1
    end

    if as_set then
      res[ind] = true
    else
      table.insert(res, ind)
    end
  end

  f:close()
  return res
end


return M
