"""Emphasize visual geometry rather than predictable editing-code boilerplate."""


def geometry_indices(pieces):
    text=''.join(pieces);spans=[]
    tag='<parameter=bbox_2d>'
    start=text.find(tag)
    if start>=0:
        start+=len(tag);end=text.find('</parameter>',start)
        if end>=0:spans.append((start,end))
    marker='polygons_roi = '
    start=text.find(marker)
    if start>=0:
        start+=len(marker);end=text.find('\n',start)
        spans.append((start,len(text)if end<0 else end))
    selected=[];position=0
    for index,piece in enumerate(pieces):
        end=position+len(piece)
        if any(position<right and end>left for left,right in spans):selected.append(index)
        position=end
    return selected


def attach_geometry_weights(processor,batch,factor):
    import torch
    labels=batch['labels']
    weights=torch.zeros_like(labels,dtype=torch.float32)
    valid=labels[0]!=-100
    positions=torch.nonzero(valid,as_tuple=False).flatten()
    ids=labels[0,valid].tolist()
    pieces=[processor.tokenizer.decode([token],skip_special_tokens=False)for token in ids]
    selected=geometry_indices(pieces)
    weights[0,positions]=1.
    if selected:
        indices=torch.tensor(selected,device=positions.device)
        weights[0,positions[indices]]=factor
    batch['token_weights']=weights
    return batch
