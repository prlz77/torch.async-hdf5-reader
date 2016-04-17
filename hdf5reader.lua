require 'hdf5'
local class = require 'class'
require 'torch'
require 'math'
--require 'cutorch'

local threads = require 'threads'
--local status, tds = pcall(require, 'tds')
--tds = status and tds or nil
--threads.Threads.serialization('threads.sharedserialize')

AsyncReader = class('AsyncReader')


-- AsyncReader is a thread asynchronous hdf5 data provider
-- @param params is a dictionary with the parameters:
--  params.nthread = number of threads (default = 1).
--  params.njob = max number of jobs (default = 1)
--  params.cuda (default = false)
--  params.batchSize = batchSize (default = 1)
--  params.postprocess = postprocessing function (default f(x) = x)
--  params.hdf5_path = path of .h5 file
--  params.data_field = database name for data. (default = 'data')
--  params.labels_field = database name for labels. (default = 'labels')
--  params.num_dim = index of the dimension containing all the images (default = 1)
--  params.h_dim = height of the image (default = 2)
--  params.w_dim = width of the image (default = 3)
--  params.cha_dim = index of the dimension containing rgb (default = 4)
--  params.shuffle = wether to shuffle data !!Overhead do not use with already shuffled data¡¡  (default = false)
function AsyncReader:__init(params)
  -- Threading variables
  self.nthread = params.nthread or 1
  self.njob = params.njob or 1
  -- Should use cuda?
  self.cuda = false or params.cuda
  -- Batch Size
  self.bs = params.batchSize or 1
  -- Postprocess
  if params.postprocess == nil then
    self.postprocess = function(x)
      return x
    end
  else
    self.postprocess = params.postprocess
  end

  -- DATA LOADING
  -- Read hdf5 datasets
  assert(params.hdf5_path ~= nil)
  self.hdf5_path = params.hdf5_path
  self.data_field = params.data_field
  self.labels_field = params.labels_field
  self.db = hdf5.open(params.hdf5_path, 'r')
  self.data = self.db:read(params.data_field or 'data')
  self.labels = self.db:read(params.labels_field or 'labels')
  -- Set internal dataset information
  self.dataSize = self.data:dataspaceSize()
  self.labelSize = self.labels:dataspaceSize()
  self.num_dim = params.num_dim or 1
  self.h_dim = params.h_dim or 2
  self.w_dim = params.w_dim or 3
  self.cha_dim = params.cha_dim or 4
  -- remove db to make it thread safe
  self.db:close()
  self.db = nil
  self.data = nil
  self.labels = nil
  -- Shuffle batch indices
  self.shuffle = false or params.shuffle
  if params.shuffle then
    print('Warning, shuffling in training time = overhead!')
    self.indexs = torch.randperm(self.dataSize[self.num_dim])
  else
    self.indexs = torch.linspace(1, self.dataSize[self.num_dim], self.dataSize[self.num_dim])
  end
  -- Point the first example
  self.dataIndex = 1
  -- Create the shared memory
  self.data_tensors = {}
  self.label_tensors = {}
  self.data_pointers = {}
  self.label_pointers = {}
  -- creates two tensors per thread: one will be filled while the other is
  -- accessed by the network
  self.rw = 1 -- which tensor we are accessing (R/W)
  for i = 1, 2 do
    self.data_tensors[i] = torch.Tensor(self.bs, self.dataSize[self.cha_dim],
                                                 self.dataSize[self.h_dim],
                                                 self.dataSize[self.w_dim])
    self.data_pointers[i] = tonumber(torch.data(self.data_tensors[i], true))
    self.label_tensors[i] = torch.Tensor(self.bs, self.labelSize[2])
    self.label_pointers[i] = tonumber(torch.data(self.label_tensors[i], true))
    -- Allocate memory in GPU
    if self.cuda then
        self.data_pointers[i]:cuda()
        self.label_pointers[i]:cuda()
    end
  end
  -- START THREADING POOL
  self.pool = threads.Threads(
    self.nthread,
	  function()
      -- Initialize the hdf5 lib
      pcall(require, 'hdf5')
      -- Share data using pointers. Two tensors needed for allowing Read while Write
      shuffle = self.shuffle
      dataSize = self.dataSize
      labelSize = self.labelSize
      num_dim = self.num_dim
      cha_dim = self.cha_dim
      w_dim = self.w_dim
      h_dim = self.h_dim
      hdf5_path = self.hdf5_path
      data_field = self.data_field or 'data'
      labels_field = self.labels_field or 'labels'
      db = hdf5.open(hdf5_path, 'r')
      data = db:read(data_field)
      labels = db:read(labels_field)
      data_tensors = {}
      label_tensors = {}
      for i = 1,2 do
        data_tensors[i] = torch.Tensor(torch.Storage(self.data_tensors[i]:view(-1):size(1), self.data_pointers[i])):view(self.data_tensors[i]:size())
        label_tensors[i] = torch.Tensor(torch.Storage(self.label_tensors[i]:view(-1):size(1), self.label_pointers[i])):view(self.label_tensors[i]:size())
      end
      indexs = self.indexs
      bs = self.bs
      -- The read/write flag
      rw = self.rw
      print('starting a new thread/state number ' .. 1)
    end
  )
  collectgarbage()
end

-- AsyncReader:fetch loads the data asynchronously
function AsyncReader:fetchData()
  local data_index = self.dataIndex
  local r_w = self.rw
  self.pool:addjob(
    function()
      rw = r_w -- self not working
      dataIndex = data_index --self not working
      -- load
      if shuffle then
        local pos = 1
        for i = dataIndex, (dataIndex + bs - 1) do
          data_tensors[rw][{{pos},{},{},{}}] = data:partial({indexs[i],indexs[i]},{1,dataSize[2]},{1,dataSize[3]},{1,dataSize[4]}):permute(num_dim,cha_dim,h_dim,w_dim)
          label_tensors[rw][{{pos},{}}] = labels:partial({indexs[i], indexs[i]}, {1,labelSize[2]})
          pos = pos + 1
        end
      else
        data_tensors[rw][{}] = data:partial({dataIndex,dataIndex+bs-1},{1,dataSize[2]},{1,dataSize[3]},{1,dataSize[4]}):permute(num_dim,cha_dim,h_dim,w_dim)
        label_tensors[rw][{}] = labels:partial({dataIndex, dataIndex+bs - 1}, {1,labelSize[2]})
      end
      data_tensors[rw] = data_tensors[rw]:contiguous()
    end
  )
  -- Increase the batch index
  self.dataIndex = self.dataIndex + self.bs
  -- If we cannot read any other entire batch, end of epoch (this code is thought for big datasets)
  if self.dataIndex + self.bs - 1 > self.dataSize[self.num_dim] then
    self.dataIndex = 1
    return true --end of epoch
  else
    return false --still in epoch
  end
end

-- AsyncReader:getNextBatch: blocking call to get the data.
function AsyncReader:getNextBatch()
    self.pool:synchronize()
    -- Put the tensor to read mode
    local rw = self.rw
    self.rw = self.rw + 1
    if self.rw > 2 then
      self.rw = 1
    end
    -- return data and labels
    return self.data_tensors[rw], self.label_tensors[rw]
end

-- AsyncReader:destructor Terminates threading pool and closes hdf5 file pointers
function AsyncReader:destructor()
  self.pool:terminate()
end
