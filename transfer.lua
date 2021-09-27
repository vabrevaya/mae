--
--  Expression transfer with multilinear model + displacements
--

local ml = require 'facemodel/multilinear'
local str = require 'common/str'



-- ============================================================================
-- CONFIG
-- ============================================================================

local CFG_geomfile = 'facemodel/geometry.off'



-- ============================================================================
-- COMMAND LINE ARGUMENTS
-- ============================================================================

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Expression transfer with multilinear model + displacements.')
cmd:text()
cmd:text('Options:')

-- command line options
cmd:option('-src',			'',		'Path to source weights (expression), as provided by mae-register.lua')
cmd:option('-target',		'',		'Path to target weights (identity), as provided by mae-register.lua')
cmd:option('-mlpath',		'',		'Path to multilinear model (directory)')
cmd:option('-outpath', 		'',  	'Output path (off file)')



-- ============================================================================
-- MAIN
-- ============================================================================

-- read arguments
cmd:text()
local params = cmd:parse(arg or {})

-- check arguments
if params.src == '' then
	cmd:error("Please provide path to source expression weights.")
end

if params.target == '' then
	cmd:error("Please provide path to target neutral weights.")
end

if params.mlpath == '' then
	cmd:error("Please provide path to multilinear model.")
end

if params.outpath == '' then
	cmd:error("Please provide output path.")
end

-- check that required files exist
if not paths.filep(params.src) then
	cmd:error("File does not exist: " .. params.src)
end

if not paths.filep(params.target) then
	cmd:error("File does not exist: " .. params.target)
end

-- make output directory
local outdir = paths.dirname(params.outpath)
paths.mkdir(outdir)

-- load model
local mlmodel = ml.MultilinearFaceModel('torch.DoubleTensor')
mlmodel:load(params.mlpath, {mean=true, geometry=CFG_geomfile})

-- load weights
local src_transf_id, src_transf_expr = mlmodel:load_weights(params.src)
local target_neutral_id, target_neutral_expr = mlmodel:load_weights(params.target)

-- reconstruct with transfered expression
local new_id = target_neutral_id:view(1, -1)
local new_expr = src_transf_expr:view(1, -1)
local newV = mlmodel:reconstruct(new_id, new_expr, true, 2)
newV = newV:squeeze()

-- save 
mlmodel:save_mesh(newV, params.outpath)

print ("Done")

