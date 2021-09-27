--
-- Multilinear model for face meshes
--

-- Don't forget to add main path in script that uses this class:
--    package.path = package.path .. ';src/?.lua'

local mlmath = require './mlmath'
local meshIO = require '../common/meshIO'
local str = require '../common/str'

--------------------------------------------------------------------------------------

local M = {}
local MultilinearFaceModel = torch.class('MultilinearFaceModel', M)

-- Constants
local ML_CORE_FILENAME = "model_core.t7"
local ML_MEAN_FILENAME = "model_mean.t7"
local ML_MEAN_ID_FILENAME = "model_mean_id.t7"
local ML_MEAN_EXPR_FILENAME = "model_mean_expr.t7"
local ML_MEAN_NEUTRAL_FILENAME = "model_mean_neutral.t7"
local ML_U2_FILENAME = "model_U2.t7"
local ML_U3_FILENAME = "model_U3.t7"
local ML_VAR2_FILENAME = "model_var2.t7"
local ML_VAR3_FILENAME = "model_var3.t7"
local ML_COV2_FILENAME = "model_cov2.t7"
local ML_COV3_FILENAME = "model_cov3.t7"
local ML_METADATA_FILENAME = "model_metadata.t7"


function MultilinearFaceModel:__init(tensorType)

	if tensorType ~= nil then
		assert(tensorType ~= 'torch.Tensor', "Be more specific about tensor type...")		-- else it doesn't work
		assert(tensorType == 'torch.DoubleTensor' or tensorType == 'torch.FloatTensor' or tensorType == 'torch.CudaTensor',
			"Unspported tensor type: " .. tensorType)

		if tensorType:sub(1, 4) == "Cuda" then
			require 'cutorch'
		end
	end

	self.tensorType = tensorType or 'torch.DoubleTensor'
	self.core = nil

	-- the following will only be loaded if specified:
	self.mean = nil
	self.mean_identity = nil
	self.mean_expression = nil
	self.neutral_mode_mean = nil
	self.faces = nil 				-- specification of geometry as a list of faces
	self.U2 = nil 					-- U matrix for mode-2
	self.U3 = nil 					-- U matrix for mode-3
	self.var2 = nil 				-- Variance for mode-2
	self.var3 = nil 				-- Variance for mode-3
	self.cov2 = nil 				-- Covariance matrix for mode-2
	self.cov3 = nil 				-- Covariance matrix for mode-3

	self.metadata = nil 			-- for any extra information we might want
end


--------------------------------------------------------------------------------------------------
--
--  Reconstruct a face mesh based on identity and expression coefficients.
--  Supports batch reconstruction.
-- 
--  Input:
--    id_weights    :  Tensor of size b x d2, with b the batch size and d2 the dimension 
--                     of the identity space. It can also receive a 1D Tensor.
--    expr_weights  :  Tensor of size b x d3, with b the batch size and d3 the dimension 
--                     of the expression space. It can also receive a 1D Tensor.
--    add_mean      :  Set to false to skip addition of mean vector.
--                     Note that the final mesh will not be correct is this is false. 
--                     Default: true.
--    vtx_format    :  Output vertex format. Options: 1 | 2 | 3. 
--                     Output Vertices will be represented as:
--                        	1: a single list of x-y-z components; or
--                        	2: a 2D tensor of size nv x 3, with each 3D point as row;
--                     Default: 1
--    res           :  If not nil, results will be put in here. Must be of appropriate size
--
--  Returns:
--    Tensor of size b x d1 (vtx_format=1), or tensor of size b x nv x 3 (vtx_format=2)
--    with the reconstructions. b=1 if input identity and expression were 1D tensors.
--
--------------------------------------------------------------------------------------------------

function MultilinearFaceModel:reconstruct(id_weights, expr_weights, add_mean, vtx_format, res)

	assert(self.core, "Core tensor was not loaded")
	assert(add_mean and self.mean or not add_mean, "Cannot add mean in reconstruction. Mean vector was not loaded")

	assert(id_weights, "No identity weights provided.")
	assert(expr_weights, "No expression weights provided.")

	assert(expr_weights:dim() == id_weights:dim(), "Identity and expression weights must have the same number of dimensions") 
	assert(id_weights:type() == expr_weights:type(), "Identity and expression weights must have the same tensor type")

	if id_weights:dim() == 2 then 
		assert(id_weights:size(1) == expr_weights:size(1), 
			"Batches of identity and expression weights must be of same size")
	end

	vtx_format = vtx_format or 1

	-- if input is 1D, convert to 1xdi tensor if we received a 1-dimensional one
	local batch_input = id_weights:dim() == 2
	
	if not batch_input then
		id_weights = id_weights:view(1,-1)
		expr_weights = expr_weights:view(1,-1)
	end

	-- (1) mode-2 multiplication with identity coefficients
	-- (2) mode-3 multiplication with expression coefficients
	
	local id_t, res_t
	local bsize = id_weights:size(1)

	-- (putting if because it's false during training, and it saves some time)
	if id_weights:type() == self.tensorType then

		id_t = mlmath.mul(self.core, 2, id_weights)
		id_t = id_t:transpose(1, 2)

		if res ~= nil then
			torch.bmm(
				res:view(res:size(1), res:size(2), 1), 
				id_t,
				expr_weights:view(expr_weights:size(1), expr_weights:size(2), 1))

			res_t = res
		else
			res_t = torch.bmm(id_t, expr_weights:view(expr_weights:size(1), expr_weights:size(2), 1))
			--res_t = mlmath.mul(id_t, 3, expr_weights)
		end

	else
		id_t = mlmath.mul(self.core, 2, id_weights:type(self.tensorType))
		id_t = id_t:transpose(1, 2)

		if res ~= nil then
			torch.bmm(
				res, 
				id_t, 
				expr_weights:type(self.tensorType):view(expr_weights:size(1), expr_weights:size(2), 1))

			res_t = res
		else
			res_t = torch.bmm(id_t, expr_weights:type(self.tensorType):view(expr_weights:size(1), expr_weights:size(2), 1))
		end
	end

	-- (3) add mean
	if add_mean then
		res_t:add(self.mean:view(1, -1):expand(bsize, self.mean:size(1)))
	end

	id_t = nil
	--collectgarbage('collect')

	if vtx_format == 2 then
		-- to change way of viewing tensor:
		--  res_t:set(storage, storageOffset, size_dim_1, stride_dim_1, size_dim_2, etc)

		if res_t:dim() == 1 then
			-- no batches => return 2D tensor
			-- size dim 1 (rows) : npoints / 3 | stride dim 1 : 3
			-- size dim 2 (cols) : 3 | stride dim 2 : 1
			res_t:set(res_t:storage(), 1, res_t:size(1)/3, 3, 3, 1)
		else
			-- with batches => return 3D tensor
			-- size dim 1 (batches) : nbatches | stride dim 1 : npoints
			res_t:set(res_t:storage(), 1, res_t:size(1), res_t:size(2), res_t:size(2)/3, 3, 3, 1)

		end
	end

	return res_t
end


--------------------------------------------------------------------------------------------------
--
--  Get matrix for the space that results from mode-n multiplying
--  the core with the identity or expression weights.
--  Only batch processing is supported for now.
--
--  Input:
--    weights   :  tensor of dimension b x d_i, with d_i the dimension of the space 
--                 for which we want the matrix (d2 or d3), and b the batch size
--
--  Returns:
--    tensor of size b x (d_1 x d_j), with d_j the dimension of the remaining space
--
--------------------------------------------------------------------------------------------------

function MultilinearFaceModel:get_space_matrix(weights)

	assert(self.core, "Core tensor was not loaded")
	assert(weights:dim() == 2, "Invalid input dimensions: expecting a 2D tensor")

	-- check dimensions of input and ensure that they're correct
	local w_dim = weights:size(2)
	local d2 = self.core:size(2)
	local d3 = self.core:size(3)

	-- TODO this won't work if both modes have the same size
	if d2 == d3 then error("Not implemented for case d2 == d3...") end
	assert( (w_dim == d2 or w_dim == d3), "Invalid input dimensions: " .. w_dim )

	-- get mode for which we need to multiply
	local nmode = (w_dim == d2) and 2 or 3
	local nmode_other = (w_dim == d2) and 3 or 2

	-- perform mode-n multiplication
	local res_tensor = mlmath.mul(self.core, nmode, weights)

	local batch_size = weights:size(1)
	local res
	
	if batch_size == 1 then
		-- squeeze the tensor into a matrix
		res = res_tensor:squeeze()
	else
		-- move the dimension with the "batch" to the first dimension
		res = res_tensor:transpose(nmode, 1)

		if nmode == 3 then
			res = res:transpose(2,3)
		end
	end

	return res
end


--------------------------------------------------------------------------------------------------
--
--  Iteratively project a mesh into identity and expression space, starting from the mean.
--  Batch processing not supported.
--  Note that the first space in which to project is arbitrary and hard-coded.
--
--  Input:
--    V        :  1D tensor of size d1, with vertex coordinates of mesh to project
--    num_its  :  number of iterations to perform
--
--  Returns:
--	  wid      :  1D tensor with identity coefficients
--	  wexpr    :  1D tensor with expression coefficients
--
--------------------------------------------------------------------------------------------------

function MultilinearFaceModel:iterative_project(V, num_its)
	assert(self.mean_identity ~= nil, "Cannot project: mean identity was not loaded")
	assert(self.mean_expression ~= nil, "Cannot project: mean expression was not loaded")

	local wid = self.mean_identity:squeeze()
	local wexpr = self.mean_expression:squeeze()

	for it = 1, num_its do
		-- project to get identity
		wid = self:project('expr', V, wexpr)

		-- project to get expression
		wexpr = self:project('id', V, wid)
	end

	return wid, wexpr
end

--------------------------------------------------------------------------------------------------
--
--  Project a mesh into either identity or expression space.
--  Batch processing not supported.
--
--  Input:
--    space  :  String. Options: 'id' | 'expr'
--    V      :  1D tensor of size d1, with vertex coordinates of mesh to project
--    w      :  1D tensor of proper dimensions with coefficients for the space
--              into which to project (i.e. size d2 if space == 'id', size d3 if space == 'expr')
--
--  Returns:
--	  outW   :  Output coefficients (size d3 if space == 'id'; size d2 if space == 'expr')
--
--------------------------------------------------------------------------------------------------

function MultilinearFaceModel:project(space, V, w)

	assert(self.core, "Core tensor was not loaded")
	assert(self.mean, "Mean vertices were not loaded")

	assert(space == 'id' or space == 'expr', "Invalid space value. Valid options: 'id' | 'expr'")
	assert(V ~= nil, "Vertex coordinates not provided")
	assert(V:dim() == 1, "Batch processing not supported (expecting 1D tensor for vertex coordinates)")
	assert(w:dim() == 1, "Batch processing not supported (expecting 1D tensor for weights)")

	-- check dimensions of input and ensure they're correct
	local d1 = self.core:size(1)
	local d2 = self.core:size(2)
	local d3 = self.core:size(3)

	assert(V:size(1) == d1, "Invalid dimensions for vertex coordinates. Expecting size " .. tostring(d1) ..
		'; got size ' .. tostring(V:size(1)))

	if space == 'id' then
		assert(w:size(1) == d2, "Invalid dimensions for w. Expecting size " .. tostring(d2) ..
			'; got size ' .. tostring(w:size(1)))
	else
		assert(w:size(1) == d3, "Invalid dimensions for w. Expecting size " .. tostring(d3) ..
			'; got size ' .. tostring(w:size(1)))
	end

	-- get matrix for the provided weights
	local M = self:get_space_matrix(w:view(1,-1))

	-- remove mean
	local tmpV = V - self.mean

	-- solve for weights
	local outW = torch.gels(tmpV, M):squeeze()

	--outV = self:reconstruct(wid:view(1,-1), wexpr:view(1,-1), true, 1):squeeze()
	return outW
end

--------------------------------------------------------------------------------------------------
--
--  Calculate the sample covariance matrix, assuming the sample mean was calculated before.
-- 
--  Input:
--    samples  :  2D tensor of size n x d, with n samples of dimension d
--                (d must correspond to the dimensions of identity or expression mode)
--    nmode    :  number of mode from which to sample (2 for identity, 3 for expression)
--
--  The covariance matrix will stored within the class. Use :save() to store results.
-- 
--------------------------------------------------------------------------------------------------

function MultilinearFaceModel:calculate_sample_cov(samples, nmode)
	assert(nmode == 2 or nmode == 3, "Invalid mode (expecting 2 or 3): " .. tostring(nmode))
	assert(samples, "Please provide samples tensor")
	assert(samples:dim() == 2, "Invalid samples: expecting 2D tensor.")
	
	local d, mean

	if nmode == 2 then
		assert(self.mean_identity ~= nil, "Cannot compute covariance matrix without mean")
		d = self:dim_id()
		mean = self.mean_identity
	else
		assert(self.mean_expression ~= nil, "Cannot compute covariance matrix without mean")
		d = self:dim_expr()
		mean = self.mean_expression
	end

	local n = samples:size(1)

	-- center samples
	for i = 1, n do
		samples[i] = samples[i]-mean
	end

	-- initialize 
	local cov = torch.Tensor(d,d):zero()

	-- iterate samples
	for i = 1, n do
		-- compute covariances on sample
		for j = 1, d do
			for k = 1, d do
				cov[j][k] = cov[j][k] + samples[i][j]*samples[i][k]
			end
		end
	end

	cov:div(n)

	-- store result
	if nmode == 2 then
		self.cov2 = cov
	else
		self.cov3 = cov
	end
end



--------------------------------------------------------------------------------------------------
-- Get functions
--------------------------------------------------------------------------------------------------

function MultilinearFaceModel:dim_points()
	-- dimension of mode 1 (3*number of vertices)
	return self.core:size(1)
end

function MultilinearFaceModel:num_vertices()
	return self.core:size(1) / 3
end

function MultilinearFaceModel:dim_id()
	return self.core:size(2)
end

function MultilinearFaceModel:dim_expr()
	return self.core:size(3)
end


--------------------------------------------------------------------------------------------------
--
--  Builds the multilinear model for the provided data tensor.
-- 
--  Input:
--    data              :  3D Tensor arranged as data x identity x expression
--    tr_identity       :  Number. Dimension to which the identity mode will be truncated.
--                         If not provided, it will not be truncated.
--    tr_expression     :  Number. Dimension to which the expression mode will be truncated.
--                         If not provided, it will not be truncated.
--    save_compactness  :  Set to true in order to measure compactness of the output model.
--                         Results will be stored in .compactness['id'] and .compactness['expr']
--                         class variables, as a 1D tensor.
--
--  The resulting model will be stored in the current class.
--  To persist the model in disk, call MultilinearFaceModel:save().
--
--  WARNING: 'data' will be modified.
--
--------------------------------------------------------------------------------------------------

function MultilinearFaceModel:build(data, tr_identity, tr_expression, save_compactness)

	assert(data:dim() == 3, "Only 3-mode tensors are supported")

	local d1 = data:size(1)
	local d2 = data:size(2)
	local d3 = data:size(3)

	assert(tr_identity == nil or tr_identity < d2, "Invalid truncated 2-mode dimensions: must be less than " .. tostring(d2))
	assert(tr_expression == nil or tr_expression < d3, "Invalid truncated 3-mode dimensions: must be less than " .. tostring(d3))

	print ("=> Building multilinear model...")

	local data_vtx_aligned = data:contiguous():view(d1, -1)		-- putting vertices as columns in a d2*d3 matrix
	
	-- 1. Calculate mean
	print ("  | centering data...")
	self.mean = torch.mean(data_vtx_aligned, 2):squeeze()
	
	-- 2. Center data
	local meanrep = self.mean:view(-1, 1):expand(d1, d2*d3)
	data_vtx_aligned:csub(meanrep)								-- in place, input was modified!
	
	local S2, S3

	-- 3. Unfold mode-2 and calculate SVD
	print ("  | mode-2 SVD...")
	self.U2, S2 = mlmath.nmode_svd(data, 2)
	S2 = S2:pow(2)
	self.var2 = S2:div(S2:sum())

	-- 4. Unfold mode-3 and calculate SVD
	print ("  | mode-3 SVD...")
	self.U3, S3 = mlmath.nmode_svd(data, 3)
	S3 = S3:pow(2)
	self.var3 = S3:div(S3:sum())

	--- before truncating, measure compactness ---
	if save_compactness then
		assert(S2:dim() == 1, 'check SVD output dimensions...')
		assert(S3:dim() == 1, 'check SVD output dimensions...')

		print ("  | measuring compactness for id...")
		self.compactness = {}
		self.compactness['id'] = torch.Tensor(S2:size(1))


		for i = 1, S2:size(1) do
			self.compactness['id'][i] = S2:sub(1, i):sum() / S2:sum()
		end

		print ("  | measuring compactness for expression...")
		self.compactness['expr'] = torch.Tensor(S3:size(1))

		for i = 1, S3:size(1) do
			self.compactness['expr'][i] = S3:sub(1, i):sum() / S3:sum()
		end
	end

	-- 5. Truncate left singular vectors
	if tr_identity ~= nil then
		-- new size: d2 x tr_identity
		self.U2 = self.U2:sub(1, -1, 1, tr_identity)
		self.var2 = self.var2:sub(1, tr_identity)
	end

	if tr_expression ~= nil then
		-- new size: d3 x tr_expression
		self.U3 = self.U3:sub(1, -1, 1, tr_expression)
		self.var3 = self.var3:sub(1, tr_expression)
	end

	-- 6. Calculate mode means based on the rows of U matrices
	self.mean_identity = torch.mean(self.U2, 1)
	self.mean_expression = torch.mean(self.U3, 1)

	-- 7. Build core:
	--    (a) Mode-2 multiplication between the data tensor and U2^T (truncated)
	--    (b) Mode-3 multiplication between previous results and U3^t (truncated)
	print ("  | building core...")
	self.core = mlmath.mul(data, 2, self.U2:t())
	self.core = mlmath.mul(self.core, 3, self.U3:t())

	print ("Done.")
end


--------------------------------------------------------------------------------------
-- MISC
--------------------------------------------------------------------------------------

function MultilinearFaceModel:type(type, tensorCache)
	assert(type, 'MultilinearFaceModel: must provide a type to convert to')

   	-- find all tensors and convert them
   	for key,param in pairs(self) do
    	self[key] = nn.utils.recursiveType(param, type, tensorCache)
   	end

	return self
end

-- 
--  Check if a set of faces (defined by vertex index)
--  corresponds to the same topology as the model
--
--  Input:
--    faces  :  2D tensor of size nf x 3, with nf the number of faces,
--              and three vertex indices on each row.
-- 
--  Output: 
--    has_same_topology  :  bool
--
function MultilinearFaceModel:has_same_topology(faces)

	local has_same_topology = (faces:size(1) == self.faces:size(1))

	if has_same_topology then
		-- further checking: that faces are indeed the same
		for ff = 1, faces:size(1) do
			for c = 1, 3 do
				has_same_topology = has_same_topology and (faces[ff][c] == self.faces[ff][c])
			end

			if not has_same_topology then break end
		end
	end

	return has_same_topology

end


--------------------------------------------------------------------------------------
-- IO
--------------------------------------------------------------------------------------

--
-- Load multilinear model from file serialized by Torch
--
function MultilinearFaceModel:load_core(core_path)
	self.core = torch.load(core_path)
end

function MultilinearFaceModel:load_mean(mean_path)
	self.mean = torch.load(mean_path)
end

function MultilinearFaceModel:load_mode_mean(identity_path, expression_path)
	self.mean_identity = torch.load(identity_path)
	self.mean_expression = torch.load(expression_path)
end

function MultilinearFaceModel:load_neutral_mean(neutral_mean_path)
	self.neutral_mean = torch.load(neutral_mean_path)
end

--
--  Stores the current state of the multilinear model in disk.
--  Each part of the model (e.g. core, mean, etc) will be saved 
--  in a different file, to allow afterwards to load only necessary information.
--  
--  Input:
--    outdir  :  Directory in which the files will be saved
--
function MultilinearFaceModel:save(outdir)

	local fullpath

	if self.core then
		fullpath = paths.concat(outdir, ML_CORE_FILENAME)
		torch.save(fullpath, self.core)
	end

	if self.mean then
		fullpath = paths.concat(outdir, ML_MEAN_FILENAME)
		torch.save(fullpath, self.mean)
	end

	if self.mean_identity then
		fullpath = paths.concat(outdir, ML_MEAN_ID_FILENAME)
		torch.save(fullpath, self.mean_identity)
	end

	if self.mean_expression then
		fullpath = paths.concat(outdir, ML_MEAN_EXPR_FILENAME)
		torch.save(fullpath, self.mean_expression)
	end

	if self.mean_neutral then
		fullpath = paths.concat(outdir, ML_MEAN_NEUTRAL_FILENAME)
		torch.save(fullpath, self.mean_neutral)
	end

	if self.U2 then
		fullpath = paths.concat(outdir, ML_U2_FILENAME)
		torch.save(fullpath, self.U2)
	end

	if self.U3 then
		fullpath = paths.concat(outdir, ML_U3_FILENAME)
		torch.save(fullpath, self.U3)
	end

	if self.var2 then
		fullpath = paths.concat(outdir, ML_VAR2_FILENAME)
		torch.save(fullpath, self.var2)
	end

	if self.var3 then
		fullpath = paths.concat(outdir, ML_VAR3_FILENAME)
		torch.save(fullpath, self.var3)
	end

	if self.cov2 then
		fullpath = paths.concat(outdir, ML_COV2_FILENAME)
		torch.save(fullpath, self.cov2)
	end

	if self.cov3 then
		fullpath = paths.concat(outdir, ML_COV3_FILENAME)
		torch.save(fullpath, self.cov3)
	end

	if self.metadata then
		fullpath = paths.concat(outdir, ML_METADATA_FILENAME)
		torch.save(fullpath, self.metadata)
	end

end

--  
--  Load multilinear model.
--  
--  Input:
--    model_path  :  Path to multilinear model directory, with files serialized by torch.
--    @extraload  :  Table with specification of extra data to load.
--                   Possible key-value pairs:
--                      + {mean = true}: to load mean face
--                      + {mode_mean = true}: to load mean for identity and expression modes
--                      + {neutral_mean = true}: to load mean expression coefficients in neutral expression
--                      + {U = true} : to load U matrices for identity and expression modes (from HOSVD)
--                      + {geometry = [path]} : to load geometry information from the specified path
--                      + {metadata = true} : to load metadaata
--
--  By default (if extraload = nil), only the core tensor is loaded. 
--
function MultilinearFaceModel:load(model_path, extraload)

	assert(model_path ~= nil, "Got nil model")
	if paths.dirp(model_path) then
		self:load_t7(model_path, extraload)
	else
		error("Other formats not implemented yet")
	end
end


--
--  Load multilinear model serialized using Torch.
--
--  Input:
--    model_dir  :  Directory where the model is located
--                  (note that each part of the model was stored in a separate file)
--    extraload  :  For options, see MultilinearFaceModel:load().
--
function MultilinearFaceModel:load_t7(model_dir, extraload)

	local fullpath = paths.concat(model_dir, ML_CORE_FILENAME)
	assert(paths.filep(fullpath), "Unable to load model: core file not found (" .. ML_CORE_FILENAME .. ")")
	self.core = torch.load(fullpath)
	self.core = self.core:type(self.tensorType)

	if extraload and extraload['mean'] then
		fullpath = paths.concat(model_dir, ML_MEAN_FILENAME)
		assert(paths.filep(fullpath), "Unable to load model mean: file not found (" .. ML_MEAN_FILENAME .. ")")
		self.mean = torch.load(fullpath):type(self.tensorType)
	end

	if extraload and extraload['mode_mean'] then
		fullpath = paths.concat(model_dir, ML_MEAN_ID_FILENAME)

		if paths.filep(fullpath) then
			self.mean_identity = torch.load(fullpath):type(self.tensorType)
			fullpath = paths.concat(model_dir, ML_MEAN_EXPR_FILENAME)

			if paths.filep(fullpath) then
				self.mean_expression = torch.load(fullpath):type(self.tensorType)
			end
		end
	end

	if extraload and extraload['neutral_mean'] then
		fullpath = paths.concat(model_dir, ML_MEAN_NEUTRAL_FILENAME)
		assert(paths.filep(fullpath), "Unable to load neutral mean: file not found (" .. ML_MEAN_NEUTRAL_FILENAME .. ")")
		self.mean_neutral = torch.load(fullpath):type(self.tensorType)
	end
		
	if extraload and extraload['U'] then
		fullpath = paths.concat(model_dir, ML_U2_FILENAME)
		assert(paths.filep(fullpath), "Unable to load U2 matrix: file not found (" .. ML_U2_FILENAME .. ")")
		self.U2 = torch.load(fullpath):type(self.tensorType)

		fullpath = paths.concat(model_dir, ML_U3_FILENAME)
		assert(paths.filep(fullpath), "Unable to load U3 matrix: file not found (" .. ML_U3_FILENAME .. ")")
		self.U3 = torch.load(fullpath):type(self.tensorType)
	end

	if extraload and extraload['var'] then
		fullpath = paths.concat(model_dir, ML_VAR2_FILENAME)

		if paths.filep(fullpath) then
			self.var2 = torch.load(fullpath):type(self.tensorType)
			fullpath = paths.concat(model_dir, ML_VAR3_FILENAME)

			if paths.filep(fullpath) then
				self.var3 = torch.load(fullpath):type(self.tensorType)
			end
		end
	end

	if extraload and extraload['cov'] then
		fullpath = paths.concat(model_dir, ML_COV2_FILENAME)
		if paths.filep(fullpath) then 
			self.cov2 = torch.load(fullpath):type(self.tensorType) 
		end

		fullpath = paths.concat(model_dir, ML_COV3_FILENAME)
		if paths.filep(fullpath) then 
			self.cov3 = torch.load(fullpath):type(self.tensorType) 
		end
	end

	if extraload and extraload['metadata'] then
		fullpath = paths.concat(model_dir, ML_METADATA_FILENAME)
		assert(paths.filep(fullpath), "Unable to load neutral mean: file not found (" .. ML_METADATA_FILENAME .. ")")
		self.metadata = torch.load(fullpath)
	end

	if extraload and extraload['geometry'] ~= nil then
		_, self.faces = meshIO.read(extraload['geometry'], false, true)
	end

end

-- 
--  Load file with geometry information.
--
--  Input:
--    mesh_file  :  path to mesh file
--
function MultilinearFaceModel:load_geometry(mesh_file)
	_, self.faces = meshIO.read(mesh_file, false, true)	-- V will be nil
end

--
--  Save list of vertices as a mesh file (off).
--  Face information must be already loaded
--
--  Input:
--    vertices  :  1D tensor of size 3v, or 2D tensor of size v x 3, 
--                 or 2D tensor of size 3 x v, with v the number of vertices.
--    outfile   :  Path to output off file
--
function MultilinearFaceModel:save_mesh(vertices, outfile)

	assert(self.faces ~= nil, "Unable to save mesh: face information was not loaded. See MultilinearFaceModel:load_geometry()")
	meshIO.save_off(vertices, self.faces, outfile)
end


--
--  Load identity and expression coefficients from a file.
--  
--  The file is expected to have the following format:
--    line 1: [dim_id] [dim_expr]
--    line 2: identity coefficients
--    line 3: expression coefficients
--        
--  Input:
--    wfile   :  path to file with weights information
--
--  Returns:
--    w_id    :  1D tensor with identity coefficients
--    w_expr  :  1D tensor with expression coefficients
--    
function MultilinearFaceModel:load_weights(wfile)

	assert(wfile, "No file given")

	local lines = {}

	for line in io.lines(wfile) do 
		table.insert(lines, line)
	end

	if #lines ~= 3 then
		error("Invalid weights file: " .. wfile)
	end
   	
   	-- first line: dimensions of identity and expression coefficients
   	local dim_parts = str.split(lines[1])
   	local dim_id = tonumber(dim_parts[1])
   	local dim_expr = tonumber(dim_parts[2])

   	-- second line: identity coefficients
   	local w_id = torch.Tensor(dim_id):type(self.tensorType)
   	local id_parts = str.split(lines[2])

	for i = 1, dim_id do
		w_id[i] = tonumber(id_parts[i])
   	end

   	-- third line: expression coefficients
   	local w_expr = torch.Tensor(dim_expr):type(self.tensorType)
   	local expr_parts = str.split(lines[3])

	for i = 1, dim_expr do
		w_expr[i] = tonumber(expr_parts[i])
   	end

	return w_id, w_expr
end

--
--  Save identity and expression coefficients in a file.
--  
--  The file will have the following format:
--    line 1: [dim_id] [dim_expr]
--    line 2: identity coefficients
--    line 3: expression coefficients
--        
--  Input:
--    savefile     :  path to file where weights will be stored
--    id_weight    :  vector with identity weights (1D tensor or 2D tensor of size 1 x d)
--    expr_weight  :  vector with expression weights (1D tensor or 2D tensor of size 1 x d)
--
function MultilinearFaceModel:save_weights(savefile, id_weight, expr_weight)

	local file = assert(io.open(savefile, 'w'))

	-- first line shows dimension of identity and expression spaces
	local id_dim_idx = id_weight:dim()
	local expr_dim_idx = expr_weight:dim()

	local dim_id = id_weight:size(id_dim_idx)
	local dim_expr = expr_weight:size(expr_dim_idx)

	file:write(string.format('%d %d\n', dim_id, dim_expr))

	-- second line contains identity coefficients
	for i = 1, dim_id do
		local val = id_weight:dim() == 1 and id_weight[i] or id_weight[1][i]
		file:write(string.format('%.6f ', val))
	end 

	file:write('\n')

	-- third line contains expression coefficients
	for i = 1, dim_expr do
		local val = expr_weight:dim() == 1 and expr_weight[i] or expr_weight[1][i]
		file:write(string.format('%.6f ', val))
	end

	file:write('\n')
	file:close()

end


return M