require 'torch'
require 'math'
require 'lfs'
require 'hdf5'

local GridSpeechBatchLoader = {}
GridSpeechBatchLoader.__index = GridSpeechBatchLoader

function GridSpeechBatchLoader.create(cqt_features, timepoints, batch_size, compile_test)
    local self = {}
    setmetatable(self, GridSpeechBatchLoader)

    self.cqt_features = cqt_features
    self.batch_size = batch_size
    self.timepoints = timepoints

    self.nspeakers = 8 -- For now
    self.nclass_speakers = 33
    self.gen_speaker = self.nclass_speakers
    self.nclass_words = 31
    self.gen_word = self.nclass_words

    local timer = torch.Timer()

    self.trainset = {}
    for spk=1, self.nspeakers do
        self.trainset[spk] = matio.load(string.format('../smcnn/grid/cqt_shariq/data/s%d.mat', spk))['X']
    end

    self.words = {
                  'blue', 'green', 'red', 'white',
                  'one', 'two', 'three', 'four', 'five',
                  'six', 'seven', 'zero',
                  'now', 'please'}

    print(string.format('data load done. Time=%.3f', timer:time().real))
    collectgarbage()
    return self
end

function GridSpeechBatchLoader:next_class_batch()
    x = torch.Tensor(self.batch_size, 1, self.cqt_features, self.timepoints)
    spk_labels  = torch.zeros(self.batch_size)
    word_labels = torch.zeros(self.batch_size)

    scale = 1

    for i=1, self.batch_size do
        spk = torch.random(1, self.nspeakers)
        word_idx = torch.random(1, #self.words)
        word = self.words[word_idx]

        word_examples = self.trainset[spk][word]

        x[{i,1,{},{}}] = word_examples[torch.random(1, word_examples:size()[1])]:mul(scale)
        spk_labels[i] = spk
        word_labels[i] = word_idx
    end

    return {x, spk_labels, word_labels}
end

return GridSpeechBatchLoader
