require 'hdf5'
require 'hdf5reader'
function test()
  -- Create fake data
  local labels = torch.linspace(1,5,5):view(5,1)
  local data = torch.Tensor(5, 10, 10, 3):zero()
  for i = 1, 5 do
    data[i]:add(i)
  end
  -- Save fake data to hdf5
  f = hdf5.open('tmp.h5', 'w')
  f:write('data', data)
  f:write('labels', labels)
  f:close()
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
  -- Instantiate hdf5 reader
  local params = {}
  params.batchSize = 2
  params.cuda = false
  params.hdf5_path = 'tmp.h5'
  params.data_field = 'data'
  params.labels_field = 'labels'
  params.num_dim = 1
  params.h_dim = 2
  params.w_dim = 3
  params.cha_dim = 4
  params.shuffle = false
  ar = AsyncReader(params)
  assert( not ar:fetchData() )
  --data:cuda()
  --labels:cuda()
  print('Checking results...')
  data2, labels2 = ar:getNextBatch()
  assert( ar:fetchData() )
  assert(torch.all(data2:eq(data[{{1,2}}])))
  assert(torch.all(labels2:eq(labels[{{1,2}}])))
  print('Test 1/4 passed')
  data2, labels2 = ar:getNextBatch()
  assert(not ar:fetchData())
  assert(torch.all(data2:eq(data[{{3,4}}])))
  assert(torch.all(labels2:eq(labels[{{3,4}}])))
  print('Test 2/4 passed successfully.')
  data2, labels2 = ar:getNextBatch()
  assert(torch.all(data2:eq(data[{{1,2}}])))
  assert(torch.all(labels2:eq(labels[{{1,2}}])))
  print('Test 3/4 passed successfully.')
  params.shuffle = true
  ar:destructor()
  ar = AsyncReader(params)
  indices = ar.indexs
  assert(not ar:fetchData())
  data2, labels2 = ar:getNextBatch()
  for i = 1, 2 do
    assert(torch.all(data2[i]:eq(data[{{indices[i]},{}}])))
    assert(torch.all(labels2[i]:eq(labels[{{indices[i]},{}}])))
  end
  print('Test 4/4 passed successfully.')
end
test()
