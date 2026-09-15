import torch



class InputEmbedding(torch.nn.Module):
    def __init__(self, *, n_features, d_model, max_context): #in features Linear(24, 32), out = d_model
        #in (batch, 8 , 24) out:(batch, 8, 32)
        super().__init__()
        self.layer_one = torch.nn.Linear(n_features, d_model)#transformers never flatten , (24,32 (wide in middle, narrow to read at end))
        self.layer_two = torch.nn.Embedding(max_context, d_model) #how many rows the table has, (8, 32)
        #one -> widens 1 tick from 24 nums to 32, no ticks see eachother, thats attention job
        #two -> table of 8 rows, x row added to wtv is in tick x, model can telll "this happened before this"
    def forward(self, x):
        indices = torch.arange(x.shape[1]) #tensor([0,1,..,7])
        #x (batch, 8 ,24)
        x = self.layer_one(x)
        #x (batch, 8 ,32)
        x = x + self.layer_two(indices)
        #emb, still 32
        return x

class CausalSelfAttention(torch.nn.Module):
    def __init__(self, *, d_model, n_heads):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("n_head must divide d_model !!")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads #head dim with 32 and 4 heads, 4 heads of 8 each
        self.q = torch.nn.Linear(d_model, d_model, bias=False)
        self.k = torch.nn.Linear(d_model, d_model, bias=False)
        self.v = torch.nn.Linear(d_model, d_model, bias=False)
        self.proj = torch.nn.Linear(d_model, d_model, bias=False) #to mix heads back into shared width

    def forward(self, x):
        batch, T, d_model = x.shape
        #each in (batch, 8 ,32)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        #split the width into heads (batch, 8, 32) -> (batch, 4, 8, 8)
        q = q.view(batch, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, T, self.n_heads, self.d_head).transpose(1, 2)
        #how much tick i want from j, in one go, whole 8x8 matrice multiply
        scores = q @ k.transpose(-2, -1) #-1 is last dimension of matrix, -2 is secnod to last
        #transposing gets .shape from (batch,8,32) to (batch,32,8)
        scores = scores / self.d_head ** 0.5 #scaling
        #tick x must NOT get in contact with tick y, x<y
        #so apply mask that makes it -inf
        """
        scores in 4 ticks                    after the mask.            1 0 0 0
                                                                        1 1 0 0
                                                                        1 1 1 0
                                                                        1 1 1 1
            t0    t1    t2    t3               t0    t1    t2    t3
        t0   2.1   0.4   1.8   0.9         t0   2.1  -inf  -inf  -inf
        t1   1.2   3.0   0.1   2.2         t1   1.2   3.0  -inf  -inf
        t2   0.7   1.1   2.4   1.6         t2   0.7   1.1   2.4  -inf
        t3   1.9   0.3   1.0   2.8         t3   1.9   0.3   1.0   2.8. 
        """
        #t0 sees itself only[1.0,0,0,0], t3 sees everything
        #choosing -inf because softmax has e^(-inf) = 0
        #but e^0 is a *heavy* weight
        mask = torch.tril(torch.ones(T, T)) #upper right traingle
        scores = scores.masked_fill(mask ==  0, float("-inf"))
        weights = scores.softmax(dim=-1) #making every row sum up to 1
        out = weights @ v #value embeddings * embedding full line
        #v_x dot... + v_y dot ...
        #stitch the heads back: (batch, 4, 8, 8) -> (batch, 8, 32)
        out = out.transpose(1, 2).contiguous().view(batch, T, d_model)
        out = self.proj(out) #new, beautifull embeddings

        return out

class FeedForward(torch.nn.Module):
    def __init__(self, *, d_model):
        super().__init__()
        self.wide = torch.nn.Linear(d_model, 4 * d_model)
        self.bend = torch.nn.GELU() #without this the two Linears collapse into one
        self.narrow = torch.nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        #no mixing across ticks, one tick at a time. attention already did the mixing
        return self.narrow(self.bend(self.wide(x)))


class Block(torch.nn.Module):
    def __init__(self, *, d_model, n_heads):
        super().__init__()
        self.norm_one = torch.nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model=d_model, n_heads=n_heads)
        self.norm_two = torch.nn.LayerNorm(d_model)
        self.feedforward = FeedForward(d_model=d_model)

    def forward(self, x):
        #x + ... is the residual: the block starts as identity and tweaks from there
        #norms go BEFORE, not after
        x = x + self.attention(self.norm_one(x))
        x = x + self.feedforward(self.norm_two(x))
        return x


class Transformer(torch.nn.Module):
    def __init__(self, *, n_features, d_model, n_heads, n_blocks, max_context):
        super().__init__()
        self.embedding = InputEmbedding(n_features=n_features, d_model=d_model,
                                        max_context=max_context)
        #a plain list hides the blocks from
        #parameters() and the optimiser never touches them, so modulelist
        self.blocks = torch.nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(Block(d_model=d_model, n_heads=n_heads))
        self.norm = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Linear(d_model, n_features)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        #only the last tick is read. the earlier ones are computed and dropped
        return self.head(x[:, -1])
