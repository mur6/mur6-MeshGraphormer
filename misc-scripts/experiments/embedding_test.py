from torch import nn
import torch
import torch.onnx

def test1():
    embedding = nn.Embedding(10, 3)
    input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
    out = embedding(input)
    print(out)

    torch.onnx.export(embedding, input, "embedding_test.onnx")

def test2():
    from src.modeling.bert import BertConfig
    from src.modeling.bert.modeling_bert import BertPreTrainedModel
    class MyTest(nn.Module):
        def __init__(self):
            super(MyTest, self).__init__()
            #self.config = config
            self.position_embeddings = nn.Embedding(512, 1024)

        def forward(self, input):
            out = self.position_embeddings(input)
            return out
    mine = MyTest()
    input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
    mine(input)
    torch.onnx.export(mine, input, "embedding_test2.onnx")

def test3():
    from src.modeling.bert import BertConfig
    from src.modeling.bert.modeling_bert import BertPreTrainedModel
    class MyTest2(nn.Module):
        def __init__(self, config):
            super(MyTest2, self).__init__()
            self.config = config
            self.position_embeddings = nn.Embedding(512, 1024)

        def forward(self, img_feats):
            #batch_size = 1
            #seq_length = 265
            #input_ids = torch.zeros([batch_size, seq_length],dtype=torch.long)

            #if position_ids is None:
            #position_ids = torch.arange(seq_length, dtype=torch.long)
            position_ids = torch.zeros([1, 265,],dtype=torch.long )#position_ids.unsqueeze(0).expand_as(input_ids)
            #print(position_ids.shape)
            out = self.position_embeddings(position_ids)
            return out

    config = BertConfig()
    mine = MyTest2(config)
    input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
    mine(input)
    torch.onnx.export(mine, input, "embedding_test3.onnx")

def test4():
    from src.modeling.bert import BertConfig
    from src.modeling.bert.modeling_graphormer import EncoderBlock
    config = BertConfig()
    config.graph_conv = False
    config.mesh_type = "hand"
    config.img_feature_dim = 512
    #'2051,512,128'
    encoder = EncoderBlock(config)

    input = torch.rand(265, 512)
    input.requires_grad_()
    #print(input)
    output = encoder(input)
    torch.onnx.export(encoder, input, "embedding_test4.onnx")
    print(output)


from src.modeling.bert.modeling_bert import BertPreTrainedModel
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


    def forward(self, img_feats, position_ids):

        #print(f"position_ids={position_ids}")
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

def test5():
    from src.modeling.bert import BertConfig
    config = BertConfig()
    config.graph_conv = False
    config.mesh_type = "hand"
    config.img_feature_dim = 512
    #'2051,512,128'
    encoder = EncoderBlock3(config)

    input = torch.rand(265, 512)
    input.requires_grad_()
    batch_size = 1#len(img_feats)
    print(batch_size)
    seq_length = 265#len(img_feats[0])
    print(seq_length)
    input_ids = torch.zeros([batch_size, seq_length],dtype=torch.long)

    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    #print(input)
    output = encoder(input, position_ids)
    torch.onnx.export(encoder, (input, position_ids), "embedding_test5.onnx")
    print(output)

test5()
