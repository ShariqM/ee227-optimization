
-- NETWORK

local CNN = {}

-- 175x140
hpsz = 3 -- Height Pool Size
wpsz = 3 -- Width Pool Size
csz = 5 -- Conv Size
ssz = 1 -- Stride Size

nchannels = {1,8,64}
full_sizes = {-1, 2048, 2048}
view_height = 17
view_width  = 13

usz = 3 -- Height Upsample Size

function CNN.adv_classifier(cqt_features, timepoints, dropout)
    local x = nn.Identity()()

    local curr = x
    for i=1, #nchannels - 1 do
        local conv = nn.SpatialConvolution(nchannels[i],nchannels[i+1],csz,csz,ssz,ssz)(curr)
        local relu = nn.ReLU()(conv)
        local pavg = nn.SpatialAveragePooling(hpsz,wpsz,hpsz,wpsz)(relu)
        curr = pavg
    end

    full_sizes[1] = nchannels[#nchannels] * view_height * view_width
    print (full_sizes[1])
    view = nn.View(full_sizes[1])(curr)

    sz_1 = 512
    sz_2 = 128
    -- Speaker
    curr = nn.Linear(full_sizes[1], sz_1)(view)
    curr = nn.ReLU()(curr)
    curr = nn.Linear(sz_1, sz_2)(curr)
    curr = nn.ReLU()(curr)
    curr = nn.Linear(sz_2, 33)(curr)
    spk_out = nn.LogSoftMax()(curr)

    -- Word
    curr = nn.Linear(full_sizes[1], sz_1)(view)
    curr = nn.ReLU()(curr)
    curr = nn.Linear(sz_1, sz_2)(curr)
    curr = nn.ReLU()(curr)
    curr = nn.Linear(sz_2, 31)(curr)
    word_out = nn.LogSoftMax()(curr)

    return nn.gModule({x}, {spk_out, word_out})
end

return CNN
