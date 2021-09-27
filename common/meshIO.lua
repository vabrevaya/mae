--
-- Input/output functions for meshes
--

local str = require './str'
local M = {}


-- ============================================================================
-- CONFIG
-- ============================================================================

-- Supported mesh extensions
local allowed_mesh_exts = {}
allowed_mesh_exts['wrl'] = true
allowed_mesh_exts['off'] = true
allowed_mesh_exts['obj'] = true


-- ============================================================================
-- READ
-- ============================================================================

--
--  Reads a mesh file and returns a tensor with vertex and/or face information.
--  
--  Input:
--    mesh_file      :  path to mesh file
--    read_vertices  :  set to false to avoid loading vertex information (default: true)
--    read_faces     :  set to false to avoid loading faces information (default: true)
--    vtx_format     :  options: 1 | 2 | 3. According to option, vertices will be stored as:
--                        1: a single list of x-y-z components;
--                        2: a 2D tensor of size nv x 3, with each 3D point as row;
--                        3: a 2D tensor of size 3 x nv, with each 3D point as column.
--                      (default: 2)
--    outV            :  optional. if set, store vertices here (it must have the proper size)
--
--  Returns:
--    V  :  1D tensor if vtx_format = 1; 2D tensor of size v x 3 if vtx_format = 2; 
--          2D tensor of size 3 x v if vtx_format = 3
--    F  :  2D tensor of size f x 3, with the indices of the face vertices on each row (1-based)
--
function M.read(mesh_file, read_vertices, read_faces, vtx_format, outV)
	
	-- default values and checks
	read_vertices = read_vertices == nil and true or read_vertices
	read_faces = read_faces == nil and true or read_faces

	assert(vtx_format == nil or vtx_format == 1 or vtx_format == 2 or vtx_format == 3, "Invalid vertex format. Allowed values:  1 | 2 | 3")
	vtx_format = vtx_format == nil and 2 or vtx_format

	-- check file extension
	local mesh_ext = paths.extname(mesh_file)

	if mesh_ext == "OFF" or mesh_ext == "off" then
		return M.read_off(mesh_file, read_vertices, read_faces, vtx_format, outV)
	elseif mesh_ext == "WRL" or mesh_ext == "wrl" then
		return M.read_wrl(mesh_file, read_vertices, read_faces, vtx_format, outV)
	elseif mesh_ext == "OBJ" or mesh_ext == "obj" then
		return M.read_obj(mesh_file, read_vertices, read_faces, vtx_format, outV)
	else
		error("Mesh extension " .. mesh_ext .. " not supported.")
	end

end

function M.read_off(mesh_file, read_vertices, read_faces, vtx_format, outV)

	-- read file
	local file = assert(io.open(mesh_file), "Unable to open mesh file: " .. mesh_file)

	-- first line should be "off"
	local line = file:read()
	line = str.strip(line)

	if line == nil or line ~= "OFF" then
		print (line)
		error("Invalid OFF file")
	end

	-- second line contains number of vertices and faces
	line = file:read()
	local lineparts = str.split(line)
	local numvertices = tonumber(lineparts[1])
	local numfaces = tonumber(lineparts[2])

	-- read vertices and store if required
	local V = nil

	if read_vertices then
		if outV ~= nil then
			-- Checked: this doesn't copy
			V = outV
		else
			if vtx_format == 1 then
				V = torch.Tensor(numvertices*3)
			elseif vtx_format == 2 then
				V = torch.Tensor(numvertices, 3)
			else
				V = torch.Tensor(3, numvertices)
			end
		end
	end

	for i = 1, numvertices do
		line = file:read()
		assert(line, "Invalid OFF file")

		if read_vertices then
			lineparts = str.split(line)
			assert(#lineparts == 3, "Invalid vertex encountered in OFF file (index " .. i .. ")")

			local v1 = lineparts[1]:gsub(',', '.')
			local v2 = lineparts[2]:gsub(',', '.')
			local v3 = lineparts[3]:gsub(',', '.')

			local v1 = tonumber(v1)
			local v2 = tonumber(v2)
			local v3 = tonumber(v3)

			if vtx_format == 1 then
				V[ (i-1)*3 + 1] = v1
				V[ (i-1)*3 + 2] = v2
				V[ (i-1)*3 + 3] = v3
			elseif vtx_format == 2 then
				V[i][1] = v1
				V[i][2] = v2
				V[i][3] = v3
			else
				V[1][i] = v1
				V[2][i] = v2
				V[3][i] = v3
			end
		end
	end

	-- read faces
	if not read_faces then
		file:close()
		return V
	end

	F = torch.Tensor(numfaces, 3)

	for i = 1, numfaces do
		line = file:read()
		assert(line, "Invalid OFF file")

		lineparts = str.split(line)
		assert(#lineparts == 4 and tonumber(lineparts[1]) == 3, "Invalid face encountered in OFF file (index " .. i .. ")")			-- expecting triangle meshes

		local v1 = tonumber(lineparts[2]) + 1		-- OFF indices are zero-based
		local v2 = tonumber(lineparts[3]) + 1
		local v3 = tonumber(lineparts[4]) + 1

		F[i][1] = v1
		F[i][2] = v2
		F[i][3] = v3
	end

	file:close()
	return V, F
end

function M.read_wrl(mesh_file, read_vertices, read_faces, vtx_format, outV)

	--- ***very basic reader***. Check:
	--- http://www.agocg.ac.uk/train/vrml2rep/part1/guide3.htm

	local file = assert(io.open(mesh_file, 'r'))
	--local float_regex = "-?%d+%.%d+"
	local float_regex = "-?%d+%(.%d+)?"
	local ind_regex = '-?%d+'
	
	local vertices = {}
	local faces = {}
	local reading_faces = false

	local vtx_points = {}
	
	while true do
		local line = file:read()
		if line == nil then break end

		if read_faces and reading_faces then
			-- check if this is a valid face specification
			local face_inds = {}

			for str_ind in line:gmatch(ind_regex) do
				table.insert(face_inds, str_ind)
			end

			if #face_inds > 3 then
				-- assuming it's a face
				-----------------------------------------------------------------------------
				-- NOTE: not checking if it's actually triangular...
				-- note that sometimes a face can be specified by more than 3 indices (...)
				-- TODO correct, check polygon_mesh_to_triangle_mesh.cpp in libigl
				-- -1 => end of indices list for this face
				-----------------------------------------------------------------------------
				if (face_inds[1] == -1 or face_inds[2] == -1 or face_inds[3] == -1) then
					error("Invalid face found: " .. line)
				end
				
				table.insert(faces, {tonumber(face_inds[1])+1, tonumber(face_inds[2])+1, tonumber(face_inds[3])+1})
			else
            	-- if not, then we assume we finished reading faces
				reading_faces = false
			end
		else
			-- check if we'll start reading faces
			local line_strip = str.strip(line)
			if line_strip:sub(1, 10) == 'coordIndex' then
				reading_faces = true
			elseif read_vertices then
				-- check if this is a vertex
				-- pattern matching to check if it's a vertex,
				-- in hope that only vertices will have this pattern
				local vtx_points = {}

				-- unfortunately this doesn't work when one of the coordinates
				-- is written as integer...
				--for str_point in line:gmatch(float_regex) do
				--	table.insert(vtx_points, str_point)
				--end
				local lineparts = str.split(line)

				if #lineparts == 3 then
					-- this *could* be a vertex (we're assuming that vertices
					-- are the only lines with three numbers)
					for i = 1, 3 do
						local point = tonumber(lineparts[i])
						if point == nil and i == 3 then
							-- maybe it has a comma in the end
							point = tonumber(lineparts[i]:sub(1, #lineparts[i]-1))
						end

						if point ~= nil then
							table.insert(vtx_points, point)
						end
					end
				end

				if #vtx_points == 3 then
					-- it's a vertex; store
					table.insert(vertices, vtx_points[1])
					table.insert(vertices, vtx_points[2])
					table.insert(vertices, vtx_points[3])
				end
			end
		end
	end

	file:close()

	-- Convert tables to tensor, with the expected format
	local V, F

	if read_vertices then
		if vtx_format == 1 then
			if outV ~= nil then 
				outV = torch.Tensor(vertices)
				V = outV
			else
				V = torch.Tensor(vertices)
			end

		else
			local numv = #vertices/3
			
			if outV == nil then
				if vtx_format == 2 then
					V = torch.Tensor(numv, 3)
				else
					V = torch.Tensor(3, numv)
				end
			else
				V = outV
			end

			for i = 1, numv do
				if vtx_format == 2 then
					V[i][1] = vertices[(i-1)*3 + 1]
					V[i][2] = vertices[(i-1)*3 + 2]
					V[i][3] = vertices[(i-1)*3 + 3]
				else
					V[1][i] = vertices[(i-1)*3 + 1]
					V[2][i] = vertices[(i-1)*3 + 2]
					V[3][i] = vertices[(i-1)*3 + 3]
				end
			end
		end
	end

	if read_faces then
		F = torch.Tensor(#faces, 3)

		for i = 1, #faces do
			F[i][1] = faces[i][1]
			F[i][2] = faces[i][2]
			F[i][3] = faces[i][3]
		end
	end

	return V, F
end

function M.read_obj(mesh_file, read_vertices, read_faces, vtx_format, outV)

	-- read file
	local file = assert(io.open(mesh_file), "Unable to open mesh file: " .. mesh_file)

	local vertices = {}
	local faces = {}

	-- read vertices and faces
	local line, parts
	while true do
		line = file:read()
		if line == nil then break end

		parts = str.split(line)

		if parts[1] == "v" and read_vertices then
			table.insert(vertices, {tonumber(parts[2]), tonumber(parts[3]), tonumber(parts[4])})
		
		elseif parts[1] == "f" and read_faces then
			-- faces:
			-- can have one of the following three formats:
            --   (1) v1 v2 v3 ...
            --   (2) v1/vt1 v2/vt2 ...           (includes vertex texture indices)
            --   (3) v1/vt1/vn1 v2/vt2/vn2 ...   (includes texture and normal indices)
            -- for now we will only care about vertex indices

            table.insert(faces, {})
            for i = 2, #parts do
            	local fparts = str.split(parts[i], '/')
            	table.insert(faces[#faces], tonumber(fparts[1]))
            end
        end 
	end
	
	-- convert to tensors
	local V, F

	if read_faces then
		F = torch.Tensor(#faces, 3)

		for i = 1, #faces do
			F[i][1] = faces[i][1]
			F[i][2] = faces[i][2]
			F[i][3] = faces[i][3]
		end
	end

	if read_vertices then
		if outV ~= nil then
			V = outV
		else

			if vtx_format == 1 then
				V = torch.Tensor(#vertices*3)
			elseif vtx_format == 2 then
				V = torch.Tensor(#vertices, 3)
			else
				V = torch.Tensor(3, #vertices)
			end
		end
	end

	for i = 1, #vertices do
		if vtx_format == 1 then
			V[ (i-1)*3 + 1] = vertices[i][1]
			V[ (i-1)*3 + 2] = vertices[i][2]
			V[ (i-1)*3 + 3] = vertices[i][3]
		elseif vtx_format == 2 then
			V[i][1] = vertices[i][1]
			V[i][2] = vertices[i][2]
			V[i][3] = vertices[i][3]
		else
			V[1][i] = vertices[i][1]
			V[2][i] = vertices[i][2]
			V[3][i] = vertices[i][3]
		end
	end

	file:close()
	return V, F
end


-- ============================================================================
-- SAVE
-- ============================================================================

--
--  Save mesh defined as a list of vertices and faces.
--
--  Input:
--    V        :  1D tensor of size 3v, or 2D tensor of size v x 3, 
--                or 2D tensor of size 3 x v, with v the number of vertices.
--    F        :  2D tensor of size f x 3, with f the number of faces,
--    outfile  :  path to output mesh file.
--                ONLY OFF OUTPUT SUPPORTED FOR NOW.
--
--  Returns:
--    true if file was saved succesfully
--
--  Note that in order to know how vertices were stored we'll check which
--  dimension is of size 3. Thus, if there are only three vertices it will
--  be confusing...
--
function M.save_off(V, F, outfile)

	assert(V ~= nil and V:dim() > 0, "Unable to save mesh: vertices not specified")
	assert(F ~= nil and F:dim() > 0, "Unable to save mesh: faces not specified")
	assert(outfile ~= nil, "Unable to save mesh: output file not specified")

	local numv
	local v_dim

	if V:dim() == 1 then
		numv = V:size(1) / 3
		v_dim = 1
	else
		v_dim = (V:size(1) == 3 and 2 or 1)
		numv = V:size(v_dim)
	end

	local numf = F:size(1)

	-- first line: "OFF"
	local file = assert(io.open(outfile, 'w'))
	file:write("OFF\n")

	-- second line: #vertices #faces #edges
	file:write(string.format('%d %d 0\n', numv, numf))

	-- write vertices
	local i = 1
	while i <= V:size(v_dim) do
		local x,y,z

		if V:dim() == 1 then
			x = V[(i-1)+1]
			y = V[(i-1)+2]
			z = V[(i-1)+3]

			i = i + 3
		else
			if v_dim == 1 then
				x = V[i][1]
				y = V[i][2]
				z = V[i][3]
			else
				x = V[1][i]
				y = V[2][i]
				z = V[3][i]
			end
			
			i = i+1
		end
		file:write(string.format('%.8f %.8f %.8f\n', x, y, z))
	end

	-- write faces
	-- (1-based in torch, converting to 0-based for off file)
	for i = 1, numf do
		file:write(string.format('3 %d %d %d\n', F[i][1] - 1, F[i][2] - 1, F[i][3] - 1))
	end

	file:close()
	return true
end


-- ============================================================================
-- MISC
-- ============================================================================

--
--  Check if a file is a supported mesh file, according to its extension
--
function M.is_mesh_file(filename)

	local ext = paths.extname(filename)

	if ext == nil then
		return false
	end

	ext = string.lower(ext)
	return (allowed_mesh_exts[ext] or false)
end



return M