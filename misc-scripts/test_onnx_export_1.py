from torch import nn
import torch
import torch.onnx
from src.modeling.bert.modeling_bert import BertPreTrainedModel

@torch.jit.script
def make_position_ids():
    batch_size = 1#len(img_feats)
    print(batch_size)
    seq_length = 265#len(img_feats[0])
    print(seq_length)
    input_ids = torch.zeros([batch_size, seq_length],dtype=torch.long)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids

class EncoderBlock3(BertPreTrainedModel):
    def __init__(self, config):
        super(EncoderBlock3, self).__init__(config)
        self.config = config
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.img_dim = config.img_feature_dim 

        try:
            self.use_img_layernorm = config.use_img_layernorm
        except:
            self.use_img_layernorm = None

        self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #if self.use_img_layernorm:
        #    self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.img_layer_norm_eps)
        #self.apply(self.init_weights)
        #self.position_ids = make_position_ids()
        #print(f"position_ids={self.position_ids}")


    def forward(self, img_feats, seq_length):
        position_ids = torch.arange(seq_length, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0)
        #position_ids = self.position_ids
        position_embeddings = self.position_embeddings(position_ids)
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Project input token features to have spcified hidden size
        img_embedding_output = self.img_embedding(img_feats)
        return position_embeddings + img_embedding_output
        # # We empirically observe that adding an additional learnable position embedding leads to more stable training
        # embeddings = position_embeddings + img_embedding_output

        # if self.use_img_layernorm:
        #     embeddings = self.LayerNorm(embeddings)
        # embeddings = self.dropout(embeddings)
        # print(embeddings)
        # return embeddings#sequence_output

class MyModule(nn.Module):
    def __init__(self, config):
        encoder1 = EncoderBlock3(config)
        encoder2 = EncoderBlock3(config)

        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([encoder1, encoder2])

    def forward(self, input, n):
        ans = []
        # ModuleList can act as an iterable, or be indexed using ints
        for _, enc in enumerate(self.linears):
            x = enc(input, n)
            ans.append(x)
        return ans

def test5():
    from src.modeling.bert import BertConfig
    config = BertConfig()
    config.graph_conv = False
    config.mesh_type = "hand"
    config.img_feature_dim = 512
    #'2051,512,128'
    model = MyModule(config)

    input = torch.rand(265, 512)
    input.requires_grad_()

    output = model(input, 265)
    torch.onnx.export(model, (input, 265), "test_onnx_export_1.onnx")
    print(output)

test5()
