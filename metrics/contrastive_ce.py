'''
Cross-entropy on cosine similarity 
'''
import torch
import dnnlib
import copy
import random
from metrics import metric_utils


def compute_cross_entropy(opts, soft_target: bool = False):
    detector_url = f"../models/gigapath.ts"
    batch_size = 256
    batch_gen = 8
    data_loader_kwargs = dict(pin_memory=True, num_workers=2, prefetch_factor=2, drop_last=True)
    assert batch_size % batch_gen == 0
    model = torch.jit.load(detector_url)
    model = model.to(opts.device).eval()

    # get truncation values
    truncation_psi = opts.G_kwargs.get('truncation_psi', 1.0)

    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)

    # Initialize.
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    num_items = len(dataset)
    
    stats = metric_utils.FeatureStats(max_items = num_items, capture_mean = True)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=0, rel_hi=1)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    random.shuffle(item_subset)

    loader = iter(torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs))

    while not stats.is_full():
        images, c = next(loader)
        G.mapping.num_broadcast = G.mapping.num_ws
        images = images.to(opts.device)

        with torch.no_grad():
            feats_real = model(images, transform=True, div255=True)

        feats_gen = []
        for c_chunk in list(torch.chunk(c, batch_size//batch_gen)):
            z = torch.randn((batch_gen, G.z_dim), device=opts.device)
            c_chunk = c_chunk.to(opts.device).to(torch.float32)
            with torch.no_grad():
                w = G.mapping(z, c_chunk, truncation_psi=truncation_psi)
                images_gen = G.synthesis(w)
                images_gen = torch.clamp((images_gen + 1)/2., 0, 1)
                feats_gen.append(model(images_gen, transform=True))

        feats_gen = torch.cat(feats_gen, dim=0)
        logits = torch.mm(feats_gen, feats_real.transpose(0,1))
        if soft_target:
            labs = torch.nn.functional.softmax(torch.mm(feats_real, feats_real.transpose(0,1)), dim=-1)
        else:
            labs = torch.arange(logits.shape[0], dtype=torch.int64).to(opts.device)

        ce = torch.nn.functional.cross_entropy(logits, labs, reduction='none')[:, None]
        stats.append_torch(ce, opts.num_gpus, opts.rank)
        progress.update(stats.num_items)

    del model
    del G
    return float(stats.get_mean())


    