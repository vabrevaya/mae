--
--  Run multilinear autoencoder on a mesh or mesh sequence
--

local files = require 'common/files'
local meshIO = require 'common/meshIO'
local ml = require 'facemodel/multilinear'

require 'nn'
require 'cunn'
require 'cudnn'
require 'image'

local mlmodel = nil 


-- ============================================================================
-- CONFIG
-- ============================================================================

-- off file with geometry information; needed to store reconstructed meshes
local CFG_geometry_file = 'facemodel/geometry.off'

-- Command to generate image from a mesh
local CFG_mesh2img_cmd = 'mesh2img/mesh2img '
--local CFG_meshcrop = true
--local CFG_mesh2img_align_template = '../facemodel/sph-ref-1.off'
local CFG_mesh2img_clean = true

-- transformations to be done by the mesh2img command
-- set to nil to ignore

-- mesh2img options
--local CFG_mesh2img_crop_size = 120
--local CFG_mesh2img_scale = 0.95
--local CFG_mesh2img_rot = {x=-24, y=-1, z=1}

-- Maximum number of reconstructions to generate at the same time
local CFG_batch_size = 99

-- Supported image extensions
local allowed_img_exts = {}
allowed_img_exts['png'] = true

-- Input image size
local img_height = 200
local img_width = 200


-- ============================================================================
-- COMMAND LINE ARGUMENTS
-- ============================================================================

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Registration of a mesh or mesh sequence with Multilinear Autoencoder.')
cmd:text()
cmd:text('Options:')

-- command line options
cmd:option('-netpath',  '',     'Path to trained network (t7 file)')
cmd:option('-inpath',   '',     'Path to mesh/image file or mesh/image sequence (folder)')
cmd:option('-outdir',   '',     'Path to output reconstructions (folder)')
cmd:option('-savew',    false,  'Optional: set in order to save output weights too.')
cmd:option('-imgdir',   '',     'Optional: Directory in which to save generated or images, or from which to load.')
cmd:option('-limit',    -1,     'Limit the number of samples to reconstruct, if running on a mesh sequence')
cmd:option('-mask',     '',     'Path to mesh that serves as mask for the sequence')

-- get options
cmd:text()
local params = cmd:parse(arg or {})

-- check options
if params.netpath == '' then
  cmd:error("Please provide path for network file")
end

if params.inpath == '' then
   cmd:error("Please specify input path (-path)")
end

if params.outdir == '' then
   cmd:error("Please specify output path (-outdir)")
end

if not paths.filep(params.netpath) or paths.extname(params.netpath) ~= 't7' then
  cmd:error("Invalid net file: " .. params.netpath)
end



-- ============================================================================
-- AUX
-- ============================================================================
--
--  Run network on a batch of (pre-computed) images.
--
--  Input:
--    net             :  Trained autoencoder, with reconstruction module at the end
--    imgs            :  Either a table with paths to input images,
--                       or a 4D tensor with the images already loaded.
--    F               :  Geometry information
--    rec_dir         :  Directory in which to save the reconstruction 
--                       (set to nil to return result without saving)
--  
--  Returns:
--    + If rec_dir is nil, returns a tensor of size [#imgs x nv], 
--      with nv the number of vertices in the model.
--    + If rec_dir ~= nil, saves in the specified directory and returns nil
--
local function _run_net(net, imgs, F, rec_dir, save_weights)

    local imgdepth = 1      -- hard-coded for now...
    local nsamples
    local input

    if not torch.isTensor(imgs) then
        nsamples = #imgs

        -- load all images into a tensor
        for i, imgpath in ipairs(imgs) do
            img = image.load(imgpath, imgdepth, 'float')        -- nChannel x height x width

            if input == nil then
                -- now that we have the image size we can initialize the tensor
                input = torch.Tensor(nsamples, imgdepth, img:size(2), img:size(3))
            end

            input[i] = img
        end
    else
        assert(imgs:dim() == 4, "Expecting a 4D tensor.")
        input = imgs
        nsamples = imgs:size(1)
    end

    -- get output weights
    input = input:cuda()

    -- TODO quick fix, it cannot handle batches of size 1...
    local doubled = false

    if input:size(1) == 1 then
        local newsize = {2}

        for i = 2, input:dim() do
            table.insert(newsize, input:size(i))
        end

        newsize = torch.LongStorage(newsize)
        input = input:expand(newsize)           -- repeated input twice
        doubled = true
    end

    local output = net:forward(input)
    local reconstruction = output[1]
    local w_id = output[2]
    local w_expr = output[3]

    if doubled then
        reconstruction = reconstruction[1]:view(1, -1)
        w_id = w_id[1]:view(1, -1)
        w_expr = w_expr[1]:view(1, -1)
    end

    if save_weights and mlmodel == nil then
        -- if saving weights we need the MultilinearModel class 
        mlmodel = ml.MultilinearFaceModel('torch.FloatTensor')
    end

    -- save reconstruction(s)
    if rec_dir == nil then
        return reconstruction
    else
        for i = 1, nsamples do
            -- mesh name will come from image name, if available
            local basename

            if not torch.isTensor(imgs) then
                basename = files.get_filename_wo_extension(imgs[i])
            else
                imgs[i] = tostring(i)
            end

            local meshname = basename .. '.off'

            local meshpath = paths.concat(rec_dir, meshname)
            meshIO.save_off(reconstruction[i], F, meshpath)

            if save_weights then
                local savefile = paths.concat(rec_dir, basename)
                mlmodel:save_weights(savefile, w_id[i], w_expr[i])
            end
        end
    end

    -- reconstruct if necessary
    if not reconstruct then
        return output
    else
        -- for now we do this sequentially because we have to split into identity and expression 
        -- coefficients, and that would duplicate the tensor if we do it in batches
        local dim_id = self.mlmodel:dim_id()
        local recs, w_id, w_expr, vertices

        if rec_dir == nil then
            recs = torch.Tensor(nsamples, self.mlmodel:dim_points())
        end

        if type(output) == 'table' then
            

        else
            -- TODO: don't do it sequentally!
            for i = 1, nsamples do

                w_id = output:sub(i, i, 1, dim_id)
                w_expr = output:sub(i, i, dim_id+1, -1)
                
                if rec_dir == nil then
                    recs[i] = self.mlmodel:reconstruct(w_id, w_expr, true):squeeze()
                else
                    vertices = self.mlmodel:reconstruct(w_id, w_expr, true):squeeze()
                    -- mesh name will come from image name, if available
                    local meshname

                    if not torch.isTensor(imgs) then
                        local filename = imgs[i]
                        meshname = files.get_filename_wo_extension(filename) .. '.off'
                    else
                        meshname = tostring(i) .. 'off'
                    end

                    local meshpath = paths.concat(rec_dir, meshname)
                    self.mlmodel:save_mesh(vertices, meshpath)
                end
            end
        end

        -- testing, for garbagecollection afterwards
        w_id = nil
        w_expr = nil
        vertices = nil

        return recs
    end
end



-- ============================================================================
-- MAIN
-- ============================================================================

print ("")
print ("---------------------")
print ("-- Running network --")
print ("---------------------")
print ("")
print ("Results will be saved in: " .. params.outdir)
print ("")

local max_samples = (params.limit > 0) and params.limit or nil

-- gather all mesh files
local meshfiles = {}

if paths.dirp(params.inpath) then
    -- mesh sequence:
    -- load valid files and sort
    for frame in paths.files(params.inpath) do

        local frame_path = paths.concat(params.inpath, frame)
        local ext = paths.extname(frame_path)
        
        if ext ~= nil then
            -- (no continue in lua...)
            ext = string.lower(ext)

            -- both mesh and images are allowed
            if meshIO.is_mesh_file(frame) or allowed_img_exts[ext] then
                table.insert(meshfiles, frame_path)
            end
        end
    end

    table.sort(meshfiles)

elseif paths.filep(params.inpath) then
    table.insert(meshfiles, params.inpath)

else
    cmd:error("Invalid mesh/mesh sequence: " .. params.inpath)
end

if #meshfiles == 0 then
    cmd:error("No input found.")
end

-- Make output directories
local imgdir = params.imgdir

if params.imgdir == '' then
    imgdir = paths.tmpname()
end

paths.mkdir(imgdir)
paths.mkdir(params.outdir)

-- Initialize network
torch.setdefaulttensortype('torch.DoubleTensor')
torch.setnumthreads(1)

local net = torch.load(params.netpath):cuda()
net:evaluate()

-- Load geometry information
_, F = meshIO.read(CFG_geometry_file, false, true, 2)

-- Generate images
print ("Generating images ...")
local mask_perc_path

for f, meshpath in ipairs(meshfiles) do

    local ext = paths.extname(meshpath)

    if allowed_img_exts[ext] == nil then
        -- it's a mesh: generate image
        local mesh_wo_ext = files.get_filename_wo_extension(meshpath)
        local imgname = mesh_wo_ext .. '.png'
        local imgpath = paths.concat(imgdir, imgname)

        if not paths.filep(imgpath) then
            --print ("  | generating image: " .. imgname)
                
            -- make command
            -- TODO: c++ code, write bindings!!
            local mesh2img_cmd = CFG_mesh2img_cmd .. ' ' .. meshpath .. ' ' .. imgpath .. 
              ' --size ' .. tostring(img_height) .. ' ' .. tostring(img_width)

            if params.mask ~= '' then
               if not paths.filep(params.mask) then
                  print ("Warning: mask does not exist: ".. params.mask .. " (ignoring)")
               else
                    mesh2img_cmd = mesh2img_cmd .. ' --mask ' .. params.mask
               end
            end

            -- transform
            if CFG_mesh2img_scale ~= nil then
                mesh2img_cmd = mesh2img_cmd .. ' --scale ' .. tostring(CFG_mesh2img_scale)
            end

            if CFG_mesh2img_rot ~= nil then
                if CFG_mesh2img_rot.x ~= nil then
                    mesh2img_cmd = mesh2img_cmd .. ' -rx ' .. tostring(CFG_mesh2img_rot.x)
                end

                if CFG_mesh2img_rot.y ~= nil then
                    mesh2img_cmd = mesh2img_cmd .. ' -ry ' .. tostring(CFG_mesh2img_rot.y)
                end

                if CFG_mesh2img_rot.z ~= nil then
                    mesh2img_cmd = mesh2img_cmd .. ' -rz ' .. tostring(CFG_mesh2img_rot.z)
                end
            end

            -- clean
            if CFG_mesh2img_clean then
                mesh2img_cmd = mesh2img_cmd .. ' --clean'
            end  

            -- crop
            -- if CFG_meshcrop then
            --   mesh2img_cmd = mesh2img_cmd .. ' --crop ' .. CFG_mesh2img_crop_size
            -- end
              --' --front --crop ' .. CFG_mesh2img_crop_size

            -- align
            -- if CFG_mesh2img_align_template ~= nil then
            --   mesh2img_cmd = mesh2img_cmd .. ' --align ' .. CFG_mesh2img_align_template
            -- end

            -- make image
            local res = sys.execute(mesh2img_cmd)
            if #res ~= 0 then error("Error while generating image: " .. res .. "\nObtained with the following command: " .. mesh2img_cmd) end
        end
    end
end

-- Run network over images and save
local num_sample = 0
local imgfiles = {}

local recs
local w_expr

print ("Reconstructing ...")

for f, meshpath in ipairs(meshfiles) do
  
    if f % CFG_batch_size == 0 then
        print (string.format('%d / %d', f, #meshfiles))
    end

    local ext = paths.extname(meshpath)

    local imgpath

    if allowed_img_exts[ext] then
        imgpath = meshpath
    else
        local mesh_wo_ext = files.get_filename_wo_extension(meshpath)
        local imgname = mesh_wo_ext .. '.png'
        imgpath = paths.concat(imgdir, imgname)
    end

    table.insert(imgfiles, imgpath)

    -- If we accumulated enough, run network and save
    if #imgfiles == CFG_batch_size or f == #meshfiles then
        _run_net(net, imgfiles, F, params.outdir, params.savew)
        imgfiles = {}
    end

    if max_samples ~= nil then
        num_sample = num_sample + 1
        if num_sample == max_samples then break end
    end
end

-- run on possible remaining image files
if #imgfiles > 0 then
    _run_net(net, imgfiles, F, params.outdir, params.savew)
end

-- remove temporary data
if params.imgdir == '' then
    paths.rmall(imgdir, 'yes')
end


print ("Done")
