--
-- Lua implementations for some tensor algebra operations, which (I think)
-- are missing in torch.
--
-- TODO: implement in C.
--


------------------------------------------------------------------

local M = {}

--
-- Mode-n multiplication  between a tensor and a matrix/vector.
--
--  Input:
--    T      : input tensor
--    nmode  : mode in which to perform multiplication.
--    A      : 2D tensor of dimensions (d x n_i), where n_i is the dimension
--             of the i-th mode (d=1 for vectors)
--
-- Output: new tensor with the same dimensions as the core, except for
-- the n-th dimension (n_i), which will be d.
--
function M.mul(T, nmode, A)

	-- check that sizes are correct for multiplication
	assert(A:dim() == 2, "Invalid input. Expecting a 2D tensor.")			-- (even if it's a vector we're expecting a 2D tensor)

	-- We will treat the core tensor as a batch of matrices, each of which
	-- will be multiplied by the vector/matrix.
	-- For batch multiplication in torch, the first dimension is the "batch" size.
	-- For the algorithm to work we need the matrix A to be contiguous. 
	-- To avoid unnecesary copying to memory, if it's not contiguous then we'll
	-- see if we can do with it's transpose.

	local res

	if A:t():isContiguous() then

		-- we'll multiply each mode-n fiber with At, _on the right_

		-- put the desired mode as the last dimension
		local aligned_T = T:transpose(nmode, 3)
		local bsize = aligned_T:size(1)

		-- for batch multiplication, create a tensor of size b * dim_mode_n * d,
		-- with repetitions of the input vectors (note that the vector is not actually 
		-- repeated in memory)

		local A_batch = A:t():view(1, A:size(2), A:size(1))
		A_batch = A_batch:expand(bsize, A:size(2), A:size(1))

		-- multiply the "batch matrices"
		local res_tr = torch.bmm(aligned_T, A_batch)

		-- put dimensions back in place
		res = res_tr:transpose(nmode, 3)
	else

		-- either we're safe with multiplying A on the left of T,
		-- (it's contiguous), or we'll have to copy to memory before doing that.
		-- in any case, we can't avoid it.

		-- put the desired mode as "column", i.e. swap nmode with the second dimension
		local aligned_T = T:transpose(nmode, 2)
		local bsize = aligned_T:size(1)

		-- for batch multiplication, create a tensor of size b * d * dim_mode_n,
		-- with repetitions of the input vectors

		local A_batch = A:contiguous():view(1, A:size(1), A:size(2))
		A_batch = A_batch:expand(bsize, A:size(1), A:size(2))

		-- multiply the "batch matrices"
		local res_tr = torch.bmm(A_batch, aligned_T)

		-- put dimensions back in place
		res = res_tr:transpose(nmode, 2)
	end

	return res
end

--
--  Calculate SVD of the tensor, unfolded in the specified mode
--  
--  Input:
--    T      :  3D tensor
--    nmode  :  number of mode in which to unfold
--
--  Returns:
--    U, S, V  : singular value decomposition of the unfolded matrix
--               A, such that A = USV'*
--

function M.nmode_svd(T, nmode)

	assert(T:dim() == 3, "Cannot handle tensors of dimension other than 3 for now...")
	assert(nmode and (nmode == 1 or nmode == 2 or nmode == 3), "Invalid mode number: " .. tostring(nmode))

	-- put all n-mode fibers as columns
	local T_unfolded = M.unfold(T, nmode)
	
	-- calculate svd of unfolded matrix
	-- NOTE: if we try to calculate svd of matrix we run out of memory
	-- So instad we'll calculate the svd of the transpose
	-- A = USV^t => A^t = VSU^t
	--U, S, V = torch.svd(T_unfolded:t(), 'S')

	if nmode == 1 then
		U, S, V = torch.svd(T_unfolded, 'S')
		return U, S, V:t()
	else
		U, S, V = torch.svd(T_unfolded:t(), 'S')
		return V, S, U:t()
	end	

	-- release memory
	-- if not was_contiguous then
	-- 	T_unfolded = nil
	-- 	collectgarbage('collect')
	-- end

end


--
--  n-mode unfolding of a 3D tensor.
--
--  Input:
--    T  :  3D Tensor 
--    nmode     :  Number of mode in which to unfold.
--    outT      :  Optional. If provided, the output tensor will be stored here.
--                 Must be of appropriate size.
--    foldtype  :  The output matrix will depend on the order by which each fiber 
--                 is traversed (i.e. the order in which free indices move).
--                 We allow two options.
--                   foldtype=1 : as in [Kolda09]
--                   foldtype=2 : as in [DeLathauwer00]
--                 Default: 1
--
--  TODO extend to more than 3 modes...
--
--  Returns:
--    New tensor of dimension 2 with the result of unfolding;
--    or nil, if outT was provided.
-- 
--  References:
--    [Kolda09] Kolda, Tamara G., and Brett W. Bader. 
--    "Tensor decompositions and applications." SIAM review 51.3 (2009): 455-500.
-- 
--    [DeLathauwer00] De Lathauwer, Lieven, Bart De Moor, and Joos Vandewalle. 
--    "A multilinear singular value decomposition." SIAM journal on Matrix Analysis and Applications 21.4 (2000): 1253-1278.
--
function M.unfold(T, nmode, outT, foldtype)

	assert(T:dim() == 3, "Expecting 3D tensor as input (got " .. tostring(T:dim() .. " dimensions)"))
	assert(nmode >= 1 and nmode <= 3, "Mode number must be between one and three")

	foldtype = foldtype or 1
	assert(foldtype == 1 or foldtype == 2, "Invalid type: " .. tostring(foldtype))

	local d1 = T:size(1)
	local d2 = T:size(2)
	local d3 = T:size(3)

	local T_unfolded = outT

	----------------

	if nmode == 3 then

		if outT == nil then
			T_unfolded =
				foldtype == 1 and (T:transpose(1,3):contiguous():view(d3,d1*d2)):clone()
				or (T:contiguous():view(d1*d2, d3):t()):clone()
		else
			if foldtype == 1 then
				outT:copy(T:transpose(1,3):contiguous():view(d3,d1*d2))
			else
				outT:copy(T:contiguous():view(d1*d2, d3):t())
			end
		end


	elseif nmode == 2 then

		-- there is no difference in type for this case
		if T_unfolded == nil then
			T_unfolded = torch.Tensor(d2, d1*d3):type(T:type())
		end

		-- this makes each "batch" represent one slice, which has to be put
		-- next to each other
		local tmpT = T:transpose(1,3)
		local ind_start = 1
		local ind_end = ind_start + d1 - 1

		for i = 1, d3 do
			T_unfolded[{ {}, {ind_start, ind_end}}] = tmpT[i]
			ind_start = ind_end+1
			ind_end = ind_start+d1-1
		end

	else

		if T_unfolded == nil then
			T_unfolded = torch.Tensor(d1, d2*d3):type(T:type())
		end

		if foldtype == 1 then
			
			-- this makes each "batch" represent one depth slice
			local tmpT = T:transpose(1,3):transpose(2,3)

			for i = 1, d3 do
				local ind_start = (i-1)*d2 + 1
				local ind_end = ind_start+d2-1
				T_unfolded[{ {}, {ind_start, ind_end}}] = tmpT[i]
			end

		else
			
			for i = 1, d3 do
				T_unfolded[i] =T[i]:contiguous():view(-1)
			end
		end

	end

	if outT == nil then
		return T_unfolded
	end

end

--
-- Fold a matrix back into a tensor.
-- Requires the original tensor dimensions (d1, d2, d3) and the mode in which it was folded.
-- The folded tensor will be of mode 3; other dimensions are not currently supported.
--
--  Input:
--    M         :  2D Tensor 
--    nmode     :  Number of mode in which to unfold.
--    d1        :  Dimension of mode-1
--    d2        :  Dimension of mode-2
--    d3        :  Dimension of mode-3
--    outT      :  Optional. If provided, the output tensor will be stored here.
--                 Must be of appropriate size.
--    foldtype  :  Specification of the order used originally for folding (see M.unfold above)
--
--  Returns:
--    New tensor of dimension 3 with the result of folding;
--    or nil, if outT was provided.
-- 
function M.fold(M, nmode, d1, d2, d3, outT, foldtype)
	assert(nmode == 1 or nmode == 2 or nmode == 3, "Invalid mode number: " .. tostring(nmode))
	
	foldtype = foldtype or 1
	assert(foldtype == 1 or foldtype == 2, "Invalid type: " .. tostring(foldtype))

	if outT ~= nil then
		-- check that dimensions are ok
		assert(outT:size(1) == d1 and outT:size(2) == d2 and outT:size(3) == d3, "Invalid dimensions for outT")
	end
	
	local T = outT == nil and torch.Tensor(d1, d2, d3):type(M:type()) or outT

	-- copy values of M into T
	local i_d1 = 1
	local i_d2 = 1
	local i_d3 = 1

	local fiber

	for i = 1, M:size(2) do

		fiber = M:t()[i]

		if nmode == 1 then
			T[{ {}, i_d2, i_d3}] = fiber

			if foldtype == 1 then
				i_d2 = i_d2+1

				if i_d2 == d2+1 then
					i_d2 = 1
					i_d3 = i_d3+1
				end
			else
				i_d3 = i_d3+1

				if i_d3 == d3+1 then
					i_d3 = 1
					i_d2 = i_d2+1
				end
			end


		elseif nmode == 2 then
			T[{ i_d1, {}, i_d3}] = fiber

			-- this is the only mode where the ordering of the columns 
			-- coincide
			i_d1 = i_d1+1

			if i_d1 == d1+1 then
				i_d1 = 1
				i_d3 = i_d3+1
			end
		else
			T[{ i_d1, i_d2, {}}] = fiber

			if foldtype == 1 then
				i_d1 = i_d1+1

				if i_d1 == d1+1 then
					i_d1 = 1
					i_d2 = i_d2+1
				end
			else
				i_d2 = i_d2+1

				if i_d2 == d2+1 then
					i_d2 = 1
					i_d1 = i_d1+1
				end
			end
		end
	end

	if outT == nil then
		return T
	end

end

return M