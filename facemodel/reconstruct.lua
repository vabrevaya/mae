--
--  Reconstruct meshes based on multilinear coefficients
--

package.path = package.path .. ';../?.lua'
local ml = require 'facemodel/multilinear'
local str = require 'common/str'


-- ============================================================================
-- CONFIG
-- ============================================================================

local CFG_geometry_file = 'geometry.off'


-- ============================================================================
-- COMMAND LINE ARGUMENTS
-- ============================================================================

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Reconstruct a set of meshes based on their multilinear coefficients.')
cmd:text()
cmd:text('Options:')

-- command line options
cmd:option('-wpath', 		'',		'Path to weights file, or directory containing weight files.')
cmd:option('-modeldir',    	'',  	'Path to multilinear model (directory).')
cmd:option('-outdir',     	'',  	'Output directory.')



-- ============================================================================
-- AUX
-- ============================================================================

--
--  Reconstruct mesh from weights file and save as off.
--
--  Input:
--    model          :  Multilinear model instance
--    inpath         :  Path to input weights file
--    outpath        :  Path to output off file
--
local function _reconstruct_and_save(model, inpath, outpath)
	
	-- TODO batch processing!
	-- load weights
	local wid, wexpr = model:load_weights(inpath)

	-- (maybe truncate) and reconstruct
	wid = wid:sub(1,model:dim_id()):view(1, -1)
	wexpr = wexpr:sub(1, model:dim_expr()):view(1, -1)
	local mesh = model:reconstruct(wid, wexpr, true, 2):squeeze()
	
	-- save mesh
	model:save_mesh(mesh, outpath)
end


-- ============================================================================
-- MAIN
-- ============================================================================

-- get options
cmd:text()
local params = cmd:parse(arg or {})

-- check options are valid
if params.wpath == '' then
	cmd:error("Please specify input path")
end

if params.modeldir == '' then
	cmd:error("Please provide path to multilinear model")
end

if params.outdir == '' then
	cmd:error("Please specify output directory")
end

if not paths.dirp(params.wpath) and not paths.filep(params.wpath) then
	cmd:error("Weights file/directory does not exist: " .. params.wdir)
end

if paths.filep(params.outdir) then
	cmd:error("Output path must be a directory")
end

if not paths.dirp(params.modeldir) then
	cmd:error("Model directory does not exist: " .. params.modelpath)
end

-- make output directory
paths.mkdir(params.outdir)

-- load multilinear model
local model = ml.MultilinearFaceModel('torch.DoubleTensor')
model:load(params.modeldir, {mean=true, geometry=CFG_geometry_file})
local dim_id = model:dim_id()
local dim_expr = model:dim_expr()

local input_is_dir = paths.dirp(params.wpath)

if input_is_dir then
	-- read files in input directory
	-- (note how we're expecting to _only_ find weight files...)

	local wdir = paths.concat(params.wpath)			-- to make it absolute

	for wfile in paths.iterfiles(wdir) do

		local fullpath = paths.concat(wdir, wfile)
		local outpath = paths.concat(params.outdir, wfile .. '.off')
		_reconstruct_and_save(model, fullpath, outpath, displ, N)
	end
else
	local fname = paths.basename(params.wpath)
	local outpath = paths.concat(params.outdir, fname .. '.off')
	_reconstruct_and_save(model, params.wpath, outpath, displ, N)
end

print ("Done. Results were saved in " .. params.outdir)