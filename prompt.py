import torch
import torch.nn as nn

class Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform', frequency_penalization=False):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.frequency_penalization = frequency_penalization

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
        
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean

        self.key_counts = torch.tensor([1 for _ in range(pool_size)])
        self.key_counts_temp = torch.tensor([1 for _ in range(pool_size)])
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None, train=False):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C
            prompt_norm = prompt_norm.to(x_embed.device)

            similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size

            # if self.frequency_penalization and train and self.key_counts.sum().item() != self.pool_size:
            if self.frequency_penalization and train and self.key_counts.sum().item() != 0:
                # key_freq = self.key_counts / self.key_counts.sum()
                # print(self.key_counts)
                # key_freq = 1 - key_freq
                # print(self.key_counts)0
                key_freq = 1/ self.key_counts
                # print(key_freq)
                key_freq = key_freq.to(x_embed.device)
                similarity = similarity.to(x_embed.device)
                similarity = similarity*key_freq
            
            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                    # Unless dimension is specified, this will be flattend if it is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                        id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                    major_prompt_id = prompt_id[major_idx] # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
            else:
                idx = prompt_mask # B, top_k

            batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

            out['prompt_idx'] = idx

            # Debugging, return sim as well
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx] # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

            out['reduce_sim'] = reduce_sim
        else:
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
        if train:
            batch_size = x_embed.shape[0]
            for i in range(batch_size):
                for j in range(self.top_k): 
                    key_id = out["prompt_idx"][i][j].item()
                    self.key_counts_temp[key_id] += 1


        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

        return out


# class ClassIncrementalPrompt(nn.Module):

#     def __init__(
#             self,
#             length=5,
#             embed_dim=768,
#             prompt_init='uniform',
#             pool_size=None,
#             top_k=None,
#             batchwise_prompt=False,
#             prompt_key_init='uniform',
#             num_classes=100,
#         ):

#         super().__init__()

#         assert pool_size % num_classes == 0, "Pool size must be a multiple of the number of classes"
#         # assert top_k <= pool_size / num_classes, "Top-k must be less or equal to the nubmer of prompt per classes"

#         self.prompt_pool = True
#         self.prompt_key = True
#         self.batchwise_prompt = batchwise_prompt
#         self.top_k = top_k
#         self.pool_size = pool_size
#         self.length = length
#         self.seen_classes = [0]
#         self.prompt_per_class = pool_size // num_classes
#         prompt_pool_shape = (pool_size, length, embed_dim)
#         if prompt_init == 'zero':
#             self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
#         elif prompt_init == 'uniform':
#             self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
#             nn.init.uniform_(self.prompt, -1, 1)
        
#         # if using learnable prompt keys
#         key_shape = (pool_size, embed_dim)
#         if prompt_key_init == 'zero':
#             self.prompt_key = nn.Parameter(torch.zeros(key_shape))
#         elif prompt_key_init == 'uniform':
#             self.prompt_key = nn.Parameter(torch.randn(key_shape))
#             nn.init.uniform_(self.prompt_key, -1, 1)

#     def l2_normalize(self, x, dim=None, epsilon=1e-12):
#         """Normalizes a given vector or matrix."""
#         square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
#         x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
#         return x * x_inv_norm
        
    
#     def forward(self, x_embed, prompt_mask=None, cls_features=None, train=False):
        
#         out = dict()

        
#         x_embed_mean = cls_features
#         seen_classes = len(self.seen_classes)
#         # low = (seen_classes-1)*self.prompt_per_classes
#         high = self.prompt_per_class*seen_classes

#         if self.top_k > high:
#             top_k = high
#         else:
#             top_k = self.top_k

#         prompt_norm = self.l2_normalize(self.prompt_key[:high], dim=1) # Pool_size, C
#         x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C
#         prompt_norm = prompt_norm.to(x_embed.device)

#         similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
        
#         _, idx = torch.topk(similarity, k=top_k, dim=1) # B, top_k
#         if self.batchwise_prompt:
#             prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
#             # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
#             # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
#             # Unless dimension is specified, this will be flattend if it is not already 1D.
#             if prompt_id.shape[0] < self.pool_size:
#                 prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
#                 id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
#             _, major_idx = torch.topk(id_counts, k=top_k) # top_k
#             major_prompt_id = prompt_id[major_idx] # top_k
#             # expand to batch
#             idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k


#         batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
#         batch_size, top_k, length, c = batched_prompt_raw.shape
#         batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

#         out['prompt_idx'] = idx

#         # Debugging, return sim as well
#         out['prompt_norm'] = prompt_norm
#         out['x_embed_norm'] = x_embed_norm
#         out['similarity'] = similarity

#         # Put pull_constraint loss calculation inside
#         batched_key_norm = prompt_norm[idx] # B, top_k, C
#         out['selected_key'] = batched_key_norm
#         x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
#         sim = batched_key_norm * x_embed_norm # B, top_k, C
#         reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

#         out['reduce_sim'] = reduce_sim
    

#         # The input with the prompt concatenated to the front. [B, prompt+token, C]
#         out['total_prompt_len'] = batched_prompt.shape[1]
#         out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

#         return out


class TaskIncrementalPrompt(nn.Module):

    def __init__(
            self,
            length=5,
            embed_dim=768,
            prompt_init='uniform',
            pool_size=None,
            top_k=None,
            batchwise_prompt=False,
            prompt_key_init='uniform',
            num_tasks=100,
        ):

        super().__init__()

        assert pool_size % num_tasks == 0, "Pool size must be a multiple of the number of classes"
        # assert top_k <= pool_size / num_classes, "Top-k must be less or equal to the nubmer of prompt per classes"

        self.prompt_pool = True
        self.prompt_key = True
        self.batchwise_prompt = batchwise_prompt
        self.top_k = top_k
        self.pool_size = pool_size
        self.length = length
        self.seen_tasks = 1
        self.num_tasks = num_tasks
        self.prompt_per_task = pool_size // num_tasks
        prompt_pool_shape = (self.prompt_per_task, length, embed_dim)
        prompt_key_shape = (self.prompt_per_task, embed_dim)

        self.prompt = nn.ParameterList([nn.Parameter(torch.zeros(prompt_pool_shape)) for t in range(num_tasks)])
        self.prompt_key = nn.ParameterList([nn.Parameter(torch.zeros(prompt_key_shape)) for t in range(num_tasks)])

        for t in range(num_tasks):
            if prompt_init == 'uniform':
                nn.init.uniform_(self.prompt[t], -1, 1)

            if prompt_key_init == 'uniform':
                nn.init.uniform_(self.prompt_key[t], -1, 1)

        self.prompt[0].requires_grad = True
        self.prompt_key[0].requires_grad = True
        for t in range(1, num_tasks):
            self.prompt[t].requires_grad = False
            self.prompt_key[t].requires_grad = False

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
        
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None, train=False, cur_task=0):
        
        out = dict()

        x_embed_mean = cls_features
        # low = (seen_classes-1)*self.prompt_per_classes
        # high = self.prompt_per_task*self.seen_tasks

        # if self.top_k > high:
        #     top_k = high
        # else:
        #     top_k = self.top_k
        top_k = self.top_k

        prompt_norm = [self.l2_normalize(self.prompt_key[t], dim=1) for t in range(self.seen_tasks)]
        # prompt_norm = self.l2_normalize(self.prompt_key[:high], dim=1) # Pool_size, C
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C
        
        prompt_norm = torch.cat(prompt_norm, dim=0)
        prompt_norm = prompt_norm.to(x_embed.device)

        similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
        
        # print(prompt_norm.shape, x_embed_mean.shape, similarity.shape)
        
        _, idx = torch.topk(similarity, k=top_k, dim=1) # B, top_k

        if self.batchwise_prompt:
            prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
            # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
            # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
            # Unless dimension is specified, this will be flattend if it is not already 1D.
            if prompt_id.shape[0] < self.pool_size:
                prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
            _, major_idx = torch.topk(id_counts, k=top_k) # top_k
            major_prompt_id = prompt_id[major_idx] # top_k
            # expand to batch
            idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k


        prompt = torch.cat(list(self.prompt.parameters()), dim=0).to(x_embed.device)
        batched_prompt_raw = prompt[idx] # B, top_k, length, C
        batch_size, top_k, length, c = batched_prompt_raw.shape
        batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C


        out['prompt_idx'] = idx

        # Debugging, return sim as well
        out['prompt_norm'] = prompt_norm
        out['x_embed_norm'] = x_embed_norm
        out['similarity'] = similarity

        # Put pull_constraint loss calculation inside
        batched_key_norm = prompt_norm[idx] # B, top_k, C
        out['selected_key'] = batched_key_norm
        x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
        sim = batched_key_norm * x_embed_norm # B, top_k, C
        reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

        out['reduce_sim'] = reduce_sim
    

        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

        return out


class ClassIncrementalPrompt(nn.Module):

    def __init__(
            self,
            length=5,
            embed_dim=768,
            prompt_init='uniform',
            pool_size=None,
            top_k=None,
            batchwise_prompt=False,
            prompt_key_init='uniform',
            num_classes=100,
        ):

        super().__init__()

        assert pool_size % num_classes == 0, "Pool size must be a multiple of the number of classes"
        # assert top_k <= pool_size / num_classes, "Top-k must be less or equal to the nubmer of prompt per classes"

        self.prompt_pool = True
        self.prompt_key = True
        self.batchwise_prompt = batchwise_prompt
        self.top_k = top_k
        self.pool_size = pool_size
        self.length = length
        self.seen_classes = list()
        self.num_classes = num_classes
        self.prompt_per_class = pool_size // num_classes
        prompt_pool_shape = (self.prompt_per_class, length, embed_dim)
        prompt_key_shape = (self.prompt_per_class, embed_dim)

        self.prompt = nn.ParameterList([nn.Parameter(torch.zeros(prompt_pool_shape)) for t in range(num_classes)])
        self.prompt_key = nn.ParameterList([nn.Parameter(torch.zeros(prompt_key_shape)) for t in range(num_classes)])

        for c in range(num_classes):
            if prompt_init == 'uniform':
                nn.init.uniform_(self.prompt[c], -1, 1)

            if prompt_key_init == 'uniform':
                nn.init.uniform_(self.prompt_key[c], -1, 1)

        self.prompt[0].requires_grad = True
        self.prompt_key[0].requires_grad = True
        for c in range(1, num_classes):
            self.prompt[c].requires_grad = False
            self.prompt_key[c].requires_grad = False


    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
        
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None, train=False, cur_task=0):
        
        out = dict()

        x_embed_mean = cls_features
        # low = (seen_classes-1)*self.prompt_per_classes
        num_seen_classes = len(self.seen_classes)
        high = self.prompt_per_class*num_seen_classes

        if self.top_k > high:
            top_k = high
        else:
            top_k = self.top_k

        prompt_norm = [self.l2_normalize(self.prompt_key[t], dim=1) for t in range(num_seen_classes)]
        # prompt_norm = self.l2_normalize(self.prompt_key[:high], dim=1) # Pool_size, C
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C
        
        prompt_norm = torch.cat(prompt_norm, dim=0)
        prompt_norm = prompt_norm.to(x_embed.device)

        similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
        
        # print(prompt_norm.shape, x_embed_mean.shape, similarity.shape)
        
        _, idx = torch.topk(similarity, k=top_k, dim=1) # B, top_k

        if self.batchwise_prompt:
            prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
            # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
            # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
            # Unless dimension is specified, this will be flattend if it is not already 1D.
            if prompt_id.shape[0] < self.pool_size:
                prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
            _, major_idx = torch.topk(id_counts, k=top_k) # top_k
            major_prompt_id = prompt_id[major_idx] # top_k
            # expand to batch
            idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k

        prompt = torch.cat(list(self.prompt.parameters()), dim=0).to(x_embed.device)
        batched_prompt_raw = prompt[idx] # B, top_k, length, C
        batch_size, top_k, length, c = batched_prompt_raw.shape
        batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C


        out['prompt_idx'] = idx

        # Debugging, return sim as well
        out['prompt_norm'] = prompt_norm
        out['x_embed_norm'] = x_embed_norm
        out['similarity'] = similarity

        # Put pull_constraint loss calculation inside
        batched_key_norm = prompt_norm[idx] # B, top_k, C
        out['selected_key'] = batched_key_norm
        x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
        sim = batched_key_norm * x_embed_norm # B, top_k, C
        reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

        out['reduce_sim'] = reduce_sim
    

        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

        return out


# class ClassIncrementalPrompt(nn.Module):

#     def __init__(
#             self,
#             length=5,
#             embed_dim=768,
#             prompt_init='uniform',
#             pool_size=None,
#             top_k=None,
#             batchwise_prompt=False,
#             prompt_key_init='uniform',
#             num_classes=100,
#         ):

#         super().__init__()

#         assert pool_size % num_classes == 0, "Pool size must be a multiple of the number of classes"
#         # assert top_k <= pool_size / num_classes, "Top-k must be less or equal to the nubmer of prompt per classes"

#         self.prompt_pool = True
#         self.prompt_key = True
#         self.batchwise_prompt = batchwise_prompt
#         self.top_k = top_k
#         self.pool_size = pool_size
#         self.length = length
#         self.seen_classes = 1
#         self.prompt_per_class = pool_size // num_classes
#         prompt_pool_shape = (pool_size, length, embed_dim)
#         if prompt_init == 'zero':
#             self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
#         elif prompt_init == 'uniform':
#             self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
#             nn.init.uniform_(self.prompt, -1, 1)
        
#         # if using learnable prompt keys
#         key_shape = (pool_size, embed_dim)
#         if prompt_key_init == 'zero':
#             self.prompt_key = nn.Parameter(torch.zeros(key_shape))
#         elif prompt_key_init == 'uniform':
#             self.prompt_key = nn.Parameter(torch.randn(key_shape))
#             nn.init.uniform_(self.prompt_key, -1, 1)

#     def l2_normalize(self, x, dim=None, epsilon=1e-12):
#         """Normalizes a given vector or matrix."""
#         square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
#         x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
#         return x * x_inv_norm
        
    
#     def forward(self, x_embed, prompt_mask=None, cls_features=None, train=False):
        
#         out = dict()

        
#         x_embed_mean = cls_features
#         # low = (seen_classes-1)*self.prompt_per_classes
#         high = self.prompt_per_class*self.seen_classes

#         if self.top_k > high:
#             top_k = high
#         else:
#             top_k = self.top_k

#         prompt_norm = self.l2_normalize(self.prompt_key[:high], dim=1) # Pool_size, C
#         x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C
#         prompt_norm = prompt_norm.to(x_embed.device)

#         similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
        
#         _, idx = torch.topk(similarity, k=top_k, dim=1) # B, top_k
#         if self.batchwise_prompt:
#             prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
#             # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
#             # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
#             # Unless dimension is specified, this will be flattend if it is not already 1D.
#             if prompt_id.shape[0] < self.pool_size:
#                 prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
#                 id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
#             _, major_idx = torch.topk(id_counts, k=top_k) # top_k
#             major_prompt_id = prompt_id[major_idx] # top_k
#             # expand to batch
#             idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k


#         batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
#         batch_size, top_k, length, c = batched_prompt_raw.shape
#         batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

#         out['prompt_idx'] = idx

#         # Debugging, return sim as well
#         out['prompt_norm'] = prompt_norm
#         out['x_embed_norm'] = x_embed_norm
#         out['similarity'] = similarity

#         # Put pull_constraint loss calculation inside
#         batched_key_norm = prompt_norm[idx] # B, top_k, C
#         out['selected_key'] = batched_key_norm
#         x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
#         sim = batched_key_norm * x_embed_norm # B, top_k, C
#         reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

#         out['reduce_sim'] = reduce_sim
    

#         # The input with the prompt concatenated to the front. [B, prompt+token, C]
#         out['total_prompt_len'] = batched_prompt.shape[1]
#         out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

#         return out


    
class TaskPrompt(Prompt):

    def __init__(
        self,
        length=5,
        embed_dim=768,
        embedding_key="mean",
        prompt_init="uniform",
        prompt_pool=False,
        prompt_key=False,
        pool_size=None,
        top_k=None,
        batchwise_prompt=False,
        prompt_key_init="uniform",
    ):
        super().__init__(
            length=length,
            embed_dim=embed_dim,
            embedding_key=embedding_key,
            prompt_init=prompt_init,
            prompt_pool=prompt_pool,
            prompt_key=prompt_key,
            pool_size=pool_size,
            top_k=top_k,
            batchwise_prompt=batchwise_prompt,
            prompt_key_init=prompt_key_init,
        )


    def forward(self, x_embed, prompt_mask=None):
        """
        Args:
            x_embed: input tensor
            prompt_mask: mask to select specific prompts.
            cls_features: key features to find the close prompts
        """
        out = dict()
        if self.prompt_pool:

            idx = torch.tensor([[p for p in range(self.pool_size)] for _ in range(x_embed.shape[0])])  # B, top_k
            batched_prompt_raw = self.prompt[idx]  # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(
                batch_size, top_k * length, c
            )  # B, top_k * length, C

            out["prompt_idx"] = idx

        out["total_prompt_len"] = batched_prompt.shape[1]
        out["prompted_embedding"] = torch.cat([batched_prompt, x_embed], dim=1)

        return out
