require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'helpers'

matio = require 'matio'
matio.use_lua_strings = true
local model_utils=require 'model_utils'
local CNN = require 'cnn'
local GridSpeechBatchLoader = require 'GridSpeechBatchLoader'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a speech conversion model')
cmd:text()
cmd:text('Options')
cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:option('-iters',200,'iterations per epoch')

cmd:option('-max_epochs',200,'number of full passes through the training data')
cmd:option('-batch_size',128,'number of sequences to train on in parallel')
cmd:option('-dropout',0,'dropout for regularization, used after each CNN hidden layer. 0 = no dropout')

cmd:option('-log',false,'Log probabilites')
cmd:option('-save_pred',false,'Save prediction')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-learning_rate',1e-2,'learning rate')
cmd:option('-learning_rate_decay',0.98,'learning rate decay')
cmd:option('-learning_rate_decay_after',20,'in number of epochs, when to start decaying the learning rate')

cmd:option('-max_epochs',200,'number of full passes through the training data')

cmd:option('-print_every',10,'how many steps/minibatches between printing out the loss')
cmd:option('-save_every',100,'Save every $1 iterations')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-seed',9415,'torch manual random number generator seed')
opt = cmd:parse(arg)

torch.manualSeed(opt.seed)

-- CUDA
if opt.type == 'float' then
    print('==> switching to floats')
    torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
    print('==> switching to CUDA')
    require 'cunn'
    torch.setdefaulttensortype('torch.FloatTensor') -- Not sure why I do this
end

-- cqt_features = 175
-- timepoints = 135
-- cqt_features = 176
-- timepoints = 83
cqt_features = 175
timepoints = 140
local loader = GridSpeechBatchLoader.create(cqt_features, timepoints, opt.batch_size, false)

init_params = true
if string.len(opt.init_from) > 0 then
    print('loading an Network from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    cnn = checkpoint.cnn
    init_params = false
else
    cnn = CNN.adv_classifier(cqt_features, timepoints, opt.dropout)
end

criterion = nn.ClassNLLCriterion()

local net_params, net_grads = cnn:parameters()

-- CUDA
if opt.type == 'cuda' then
   cnn:cuda()
   criterion:cuda()
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(cnn)
nparams = params:nElement()
print('number of parameters in the model: ' .. nparams)

if init_params then
    params:normal(-1/torch.sqrt(nparams), 1/torch.sqrt(nparams))
    params:uniform(-0.08, 0.08) -- small uniform numbers
end

first = true
last_grad_params = params:clone() -- Dummy
last_params = params:clone() -- Dummy

function feval(p)
    if p ~= params then
        params:copy(p)
    end
    grad_params:zero()

    local perf = false
    local timer = torch.Timer()
    x, spk_labels, word_labels = unpack(loader:next_class_batch(train))

    if perf then print (string.format("Time 1: %.3f", timer:time().real)) end

    if opt.type == 'cuda' then
        x = x:float():cuda()
        spk_labels  =  spk_labels:float():cuda()
        word_labels = word_labels:float():cuda()
    end

    if perf then print (string.format("Time 2: %.3f", timer:time().real)) end

    spk_pred, word_pred = unpack(cnn:forward(x))
    if opt.log then
        for k=1, 8 do
            i = torch.random(1, opt.batch_size)
            sprob = torch.exp(spk_pred)[{i,spk_labels[i]}]
            wprob = torch.exp(word_pred)[{i,word_labels[i]}]
            print (string.format("\tP(S=%d)=%.2f || P(W)=%.2f", spk_labels[i], sprob, wprob))
        end
    end

    if perf then print (string.format("Time 3: %.3f", timer:time().real)) end
    local loss = criterion:forward(spk_pred, spk_labels)
    loss = loss + criterion:forward(word_pred, word_labels)
    if perf then print (string.format("Time 4: %.3f", timer:time().real)) end

    doutput_spk  = criterion:backward(spk_pred,  spk_labels):float()
    doutput_word = criterion:backward(word_pred, word_labels):float()
    if opt.type == 'cuda' then doutput_spk  = doutput_spk:cuda() end
    if opt.type == 'cuda' then doutput_word = doutput_word:cuda() end

    cnn:backward(x, {doutput_spk, doutput_word})

    if not first then
        grad_diff = grad_params - last_grad_params
        param_diff = params - last_params
        L = torch.norm(grad_diff) / torch.norm(param_diff)
        m = grad_diff:dot(param_diff) / torch.norm(param_diff) ^ 2
        print (string.format("1/L:%.5f || 2/(L+m):%.5f || m:%.3f", 1/L, 2/(L+m), m))
    end
    first = false
    last_params:copy(params)
    last_grad_params:copy(grad_params)
    -- print ("Params", torch.norm(params))

    return loss, grad_params
end

local iterations = opt.max_epochs * opt.iters
local iterations_per_epoch = opt.iters
local loss0 = nil
local optim_state = {learningRate = opt.learning_rate}
T = 100
decay_time = T

for i = 1, iterations do
    local epoch = i / iterations_per_epoch

    local timer = torch.Timer()
    local _, loss = optim.sgd(feval, params, optim_state)
    local time = timer:time().real

    loss = loss[1]

    if i >= decay_time then
        optim_state.learningRate = optim_state.learningRate * 1./2
        print(string.format("Decayed learning rate to %.5f", optim_state.learningRate))
        T = 2*T
        decay_time = i + T
    end

    if i == 1 or i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, loss, grad_params:norm() / params:norm(), time))
    end

    if (i % opt.save_every == 0 or i == iterations) then
        local savefile = string.format('%s/net_classify_%.2f.t7', opt.checkpoint_dir, epoch)
        print('saving checkpoint to ' .. savefile)
        checkpoint = {}
        checkpoint.cnn = cnn
        torch.save(savefile, checkpoint)
        print('saved checkpoint to ' .. savefile)
    end


    -- handle early stopping if things are going really bad
    if loss ~= loss then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then loss0 = loss end
end
