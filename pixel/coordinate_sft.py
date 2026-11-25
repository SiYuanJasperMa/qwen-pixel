"""Make coordinate units explicit without changing the teacher's rendered pixels."""
import ast

NORMALIZED_POINT_FUNCTION = '''def _px(coords):
    height, width = pixels.shape[:2]
    if coords and isinstance(coords[0], (list, tuple)):
        return [(round(float(x)*width/1000, 6), round(float(y)*height/1000, 6)) for x,y in coords]
    return [round(float(value)*(width if i%2==0 else height)/1000, 6) for i,value in enumerate(coords)]
'''
ROI_POINT_FUNCTION = '''import json
_roi = json.load(open('region.json'))['source_bbox_pixels']
def _px(coords):
    x0,y0,x1,y1 = _roi
    width,height = x1-x0,y1-y0
    if coords and isinstance(coords[0], (list, tuple)):
        return [(round(x0+float(x)*width/1000,6), round(y0+float(y)*height/1000,6)) for x,y in coords]
    return [round((x0 if i%2==0 else y0)+float(value)*(width if i%2==0 else height)/1000,6) for i,value in enumerate(coords)]
def _span(coords):
    x0,y0,x1,y1 = _roi
    return [round(float(value)*((x1-x0) if i%2==0 else (y1-y0))/1000,6) for i,value in enumerate(coords)]
'''


def numeric_coordinates(node):
    try:
        value=ast.literal_eval(node)
    except (ValueError,TypeError):
        return None
    if not isinstance(value,(tuple,list)) or not value:
        return None
    flat=value if not isinstance(value[0],(tuple,list)) else [v for pair in value for v in pair]
    if not all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in flat):
        return None
    return value


def normalized_node(value,width,height,origin=(0,0)):
    if isinstance(value[0],(tuple,list)):
        scaled=[(round((x-origin[0])*1000/width,8),round((y-origin[1])*1000/height,8))for x,y in value]
    else:
        scaled=[round((v-origin[i%2])*1000/(width if i%2==0 else height),8)for i,v in enumerate(value)]
    return ast.parse(repr(scaled),mode='eval').body


def normalize_drawing_code(code,width,height,frame=None):
    origin=(0,0)
    if frame is not None:
        origin=tuple(frame[:2]);width,height=frame[2]-frame[0],frame[3]-frame[1]
        if width<=0 or height<=0:raise ValueError('ROI must have positive dimensions.')
    tree=ast.parse(code)
    names=set()
    for node in ast.walk(tree):
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                and node.func.attr in ('polygon','ellipse','rectangle') and node.args
                and isinstance(node.args[0],ast.Name)):
            names.add(node.args[0].id)
    bindings={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            value=numeric_coordinates(node.value)
            if name in names and value is not None:
                bindings[name]=value
    count=0
    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self,node):
            nonlocal count
            self.generic_visit(node)
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in bindings:
                value=numeric_coordinates(node.value)
                if value is not None:node.value=normalized_node(value,width,height,origin)
            if len(node.targets)==1 and isinstance(node.targets[0],(ast.Tuple,ast.List)):
                target=node.targets[0]
                value=numeric_coordinates(node.value)
                # Known x/y centers and radii used by numeric ellipse masks.
                if (len(target.elts)==2 and all(isinstance(t,ast.Name)for t in target.elts)
                        and (target.elts[0].id,target.elts[1].id) in
                        (('cx','cy'),('rx','ry'),('hx','hy'),('hrx','hry'))
                        and value is not None and len(value)==2):
                    radius=target.elts[0].id in ('rx','hrx')
                    node.value=ast.Call(func=ast.Name(id='_span'if frame is not None and radius else '_px',ctx=ast.Load()),
                        args=[normalized_node(value,width,height,(0,0)if radius else origin)],keywords=[])
                    count+=1
            return node
        def visit_Call(self,node):
            nonlocal count
            self.generic_visit(node)
            if (isinstance(node.func,ast.Attribute) and node.func.attr in
                    ('polygon','ellipse','rectangle') and node.args):
                first=node.args[0]
                value=numeric_coordinates(first)
                if value is not None:
                    node.args[0]=ast.Call(func=ast.Name(id='_px',ctx=ast.Load()),
                        args=[normalized_node(value,width,height,origin)],keywords=[])
                    count+=1
                elif isinstance(first,ast.Name)and first.id in bindings:
                    node.args[0]=ast.Call(func=ast.Name(id='_px',ctx=ast.Load()),args=[first],keywords=[])
                    count+=1
            return node
    tree=ast.fix_missing_locations(Rewrite().visit(tree))
    rewritten=ast.unparse(tree)
    helper=ROI_POINT_FUNCTION if frame is not None else NORMALIZED_POINT_FUNCTION
    return (helper+'\n'+rewritten if count else rewritten),count
