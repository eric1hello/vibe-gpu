
import ast
import inspect
import sys
import os
import math
from tools.assembler import Assembler

class RegAllocator:
    def __init__(self):
        self.free_regs = list(range(1, 32)) # Upgraded to 32 Regs
        self.var_map = {} 
        self.temp_map = {} 

    def alloc(self, name=None):
        if not self.free_regs:
            raise RuntimeError(f"Out of registers! Used: {self.var_map}")
        reg = self.free_regs.pop(0)
        if name:
            self.var_map[name] = reg
        return reg

    def get(self, name):
        return self.var_map.get(name)

    def free(self, reg):
        if reg != 0 and reg not in self.free_regs:
            self.free_regs.insert(0, reg)
            self.free_regs.sort()

    def free_var(self, name):
        if name in self.var_map:
            reg = self.var_map[name]
            del self.var_map[name]
            self.free(reg)

class CudaCompiler(ast.NodeVisitor):
    def __init__(self, asm, arg_values, block_dim):
        self.asm = asm
        self.allocator = RegAllocator()
        self.arg_values = arg_values
        self.block_dim = block_dim
        self.label_counter = 0
        
        # Pre-calculate log2 for shift
        if not math.log2(block_dim).is_integer():
            raise ValueError("Block Dim must be power of 2 for this Vibe GPU")
        self.block_shift = int(math.log2(block_dim))
        self.block_mask = block_dim - 1

    def new_label(self, prefix="L"):
        self.label_counter += 1
        return f"{prefix}_{self.label_counter}"

    def compile(self, func):
        source = inspect.getsource(func)
        import textwrap
        source = textwrap.dedent(source)
        tree = ast.parse(source)
        self.visit(tree)

    def load_large_const(self, reg, val):
        chunk = val
        if chunk > 1023: chunk = 1023 # ISA updated: 11 bit imm (-1024 to 1023)
        if chunk < -1024: chunk = -1024
        
        self.asm.ldi(reg, chunk)
        current_val = chunk
        
        while current_val != val:
            diff = val - current_val
            chunk = diff
            if chunk > 1023: chunk = 1023
            if chunk < -1024: chunk = -1024
            
            tmp = self.allocator.alloc()
            self.asm.ldi(tmp, chunk)
            self.asm.add(reg, reg, tmp)
            self.allocator.free(tmp)
            current_val += chunk

    # --- ID Calculation Helpers ---
    def emit_hw_id(self, dest_reg):
        """
        Calculates Global Hardware ID:
        HW_ID = SMID * 32 + WARPID * 8 + TID
        """
        r_temp = self.allocator.alloc()
        
        # 1. Warp Offset = WARPID * 8
        self.asm.warpid(dest_reg)
        self.asm.ldi(r_temp, 3) # Shift 3 bits (x8)
        self.asm.sll(dest_reg, dest_reg, r_temp)
        
        # 2. Add TID
        self.asm.tid(r_temp)
        self.asm.add(dest_reg, dest_reg, r_temp)
        
        # 3. Add SM Offset = SMID * 32
        self.asm.smid(r_temp)
        r_shift = self.allocator.alloc()
        self.asm.ldi(r_shift, 5) # Shift 5 bits (x32)
        self.asm.sll(r_temp, r_temp, r_shift)
        self.allocator.free(r_shift)
        
        self.asm.add(dest_reg, dest_reg, r_temp)
        self.allocator.free(r_temp)

    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            arg_name = arg.arg
            val = self.arg_values.get(arg_name, 0)
            reg = self.allocator.alloc(arg_name)
            self.load_large_const(reg, val)
        
        for stmt in node.body:
            self.visit(stmt)
        self.asm.halt()


    def visit_AugAssign(self, node):
        # Support +=
        target = node.target.id
        reg_target = self.allocator.get(target)
        reg_val = self.visit_expr(node.value)
        if isinstance(node.op, ast.Add):
            self.asm.add(reg_target, reg_target, reg_val)
        self.free_if_temp(reg_val)

    def free_if_temp(self, reg):
        is_bound = False
        for var_reg in self.allocator.var_map.values():
            if var_reg == reg:
                is_bound = True
                break
        if not is_bound:
            self.allocator.free(reg)

    def visit_Compare(self, node):
        # Handle if x == y
        left = self.visit_expr(node.left)
        right = self.visit_expr(node.comparators[0])
        
        # We only have BEQ/BNE at assembly level.
        # But here we need to return a boolean-like value (0 or 1) or a difference
        # for the BEQ in visit_If to consume.
        # Let's return (left - right). If 0, they are equal.
        
        # For proper boolean semantics (x == y -> 1, else 0):
        # This is harder with current ISA. 
        # Hack: For visit_If, it just checks if cond == 0.
        # So if we return (left - right), 'If' will see 0 as Equal, Non-Zero as Not Equal.
        
        # Wait, visit_If logic:
        # if node.orelse:
        #    self.asm.beq(cond, zero, lbl_else) 
        # This means if cond == 0, it jumps to ELSE.
        # So cond must be TRUE (Non-Zero) to execute THEN.
        
        # If input code is: if gid == 0:
        # We want: If Equal, execute THEN. 
        # So cond should be TRUE if Equal?
        # (gid - 0) is 0 if Equal. So cond is 0.
        # BEQ(0, 0, else) -> Jumps to Else. WRONG.
        
        # So for `==`, we want cond to be 1 if Equal.
        # Or, we can make visit_If smarter.
        
        # Let's support specific Compare ops in visit_If?
        # No, visit_If calls visit_expr.
        
        # Let's implement Compare to return (left == right).
        # res = (left == right)
        # diff = left - right
        # if diff == 0: res = 1 else res = 0
        
        res = self.allocator.alloc()
        diff = self.allocator.alloc()
        self.asm.sub(diff, left, right)
        
        # Convert diff to boolean
        # If diff == 0, res = 1
        # If diff != 0, res = 0
        
        # Logic:
        # res = 1
        # BEQ diff, 0, skip
        # res = 0
        # skip:
        
        lbl_true = self.new_label("CMP_TRUE")
        lbl_end = self.new_label("CMP_END")
        zero = self.allocator.alloc()
        self.asm.ldi(zero, 0)
        
        if isinstance(node.ops[0], ast.Eq):
            self.asm.ldi(res, 0)          # Default False
            self.asm.beq(diff, zero, lbl_true) # If Diff==0, Jump True
            self.asm.jump(lbl_end)
            
            self.asm.label(lbl_true)
            self.asm.ldi(res, 1)          # True
            self.asm.label(lbl_end)
            
        elif isinstance(node.ops[0], ast.NotEq):
            self.asm.ldi(res, 1)          # Default True
            self.asm.beq(diff, zero, lbl_true) # If Diff==0 (Equal), Jump False logic
            self.asm.jump(lbl_end)
            
            self.asm.label(lbl_true)
            self.asm.ldi(res, 0)          # False
            self.asm.label(lbl_end)
            
        self.allocator.free(zero)
        self.allocator.free(diff)
        self.free_if_temp(left)
        self.free_if_temp(right)
        return res

    def visit_expr(self, node):
        if isinstance(node, ast.BinOp):
            left = self.visit_expr(node.left)
            right = self.visit_expr(node.right)
            res = self.allocator.alloc()
            if isinstance(node.op, ast.Add): self.asm.add(res, left, right)
            elif isinstance(node.op, ast.Sub): self.asm.sub(res, left, right)
            elif isinstance(node.op, ast.Mult): self.asm.mul(res, left, right)
            elif isinstance(node.op, ast.LShift): self.asm.sll(res, left, right)
            elif isinstance(node.op, ast.RShift): self.asm.srl(res, left, right)
            elif isinstance(node.op, ast.BitAnd): self.asm.and_(res, left, right)
            elif isinstance(node.op, ast.BitOr): self.asm.or_(res, left, right)
            self.free_if_temp(left); self.free_if_temp(right)
            return res
        elif isinstance(node, ast.Name):
            return self.allocator.get(node.id)
        elif isinstance(node, ast.Constant):
            res = self.allocator.alloc()
            self.load_large_const(res, node.value)
            return res
        elif isinstance(node, ast.Attribute):
            return self.handle_cuda_attr(node)
        elif isinstance(node, ast.Call):
            return self.handle_call(node)
        elif isinstance(node, ast.Subscript):
            # Handle Load: x = a[i]
            return self.visit_Subscript(node)
        elif isinstance(node, ast.Compare):
            return self.visit_Compare(node)
        else:
            raise NotImplementedError(f"Expr {type(node)}")

    def handle_cuda_attr(self, node):
        # Handle cuda.threadIdx.x, cuda.blockIdx.x, cuda.blockDim.x
        # node.value is Attribute(cuda.threadIdx) or Name(cuda)
        # This is getting complex parsing A.B.C
        # Flatten the name
        full_name = self.get_full_attr_name(node)
        
        if full_name == "cuda.blockDim.x":
            res = self.allocator.alloc()
            self.load_large_const(res, self.block_dim)
            return res
            
        elif full_name == "cuda.threadIdx.x":
            # HW_ID & (BlockDim - 1)
            hw_id = self.allocator.alloc()
            self.emit_hw_id(hw_id)
            
            mask_reg = self.allocator.alloc()
            self.load_large_const(mask_reg, self.block_mask)
            
            self.asm.and_(hw_id, hw_id, mask_reg)
            self.allocator.free(mask_reg)
            return hw_id
            
        elif full_name == "cuda.blockIdx.x":
            # HW_ID >> log2(BlockDim)
            hw_id = self.allocator.alloc()
            self.emit_hw_id(hw_id)
            
            shift_reg = self.allocator.alloc()
            self.load_large_const(shift_reg, self.block_shift)
            
            self.asm.srl(hw_id, hw_id, shift_reg)
            self.allocator.free(shift_reg)
            return hw_id
            
        elif full_name == "cuda.gridDim.x":
             # Total HW Threads / Block Dim
             total_threads = 128
             grid_dim = total_threads // self.block_dim
             res = self.allocator.alloc()
             self.load_large_const(res, grid_dim)
             return res
             
        raise ValueError(f"Unknown CUDA Attr: {full_name}")

    def get_full_attr_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self.get_full_attr_name(node.value) + "." + node.attr
        return ""

    def handle_call(self, node):
        name = self.get_full_attr_name(node.func)
        if name == "cuda.grid":
            res = self.allocator.alloc()
            self.emit_hw_id(res)
            return res
        elif name == "cuda.syncthreads":
            # Simple JOIN based sync (not real barrier yet)
            self.asm.join()
            res = self.allocator.alloc()
            self.asm.ldi(res, 0)
            return res
        elif name == "cuda.csr_cycle":
            # Read CSR Cycle (0)
            res = self.allocator.alloc()
            op_a_reg = self.allocator.alloc()
            self.asm.ldi(op_a_reg, 0)
            self.asm.csr(res, op_a_reg)
            self.allocator.free(op_a_reg)
            return res
        elif name == "cuda.csr_instret":
            # Read CSR InstRet (1)
            res = self.allocator.alloc()
            op_a_reg = self.allocator.alloc()
            self.asm.ldi(op_a_reg, 1)
            self.asm.csr(res, op_a_reg)
            self.allocator.free(op_a_reg)
            return res
            
        raise NotImplementedError(f"Call {name}")

    def visit_If(self, node):
        cond = self.visit_expr(node.test)
        lbl_else = self.new_label("ELSE")
        lbl_end = self.new_label("END")
        zero = self.allocator.alloc()
        self.asm.ldi(zero, 0)
        
        # If Eq: BNE to Else; If NotEq: BEQ to Else
        # Simple check: if cond == 0, jump else
        
        if node.orelse:
            self.asm.beq(cond, zero, lbl_else)
            
            # Then
            for s in node.body: self.visit(s)
            self.asm.jump(lbl_end)
            
            # Else
            self.asm.label(lbl_else)
            for s in node.orelse: self.visit(s)
            
            self.asm.label(lbl_end)
            self.asm.join()
        else:
            self.asm.beq(cond, zero, lbl_end)
            
            # Then
            for s in node.body: self.visit(s)
            
            self.asm.label(lbl_end)
            self.asm.join()
            
        self.allocator.free(zero)
        self.free_if_temp(cond)

    def visit_For(self, node):
        # range loop support
        iter_var = node.target.id
        iter_reg = self.allocator.alloc(iter_var)
        
        args = node.iter.args
        start = 0
        step = 1
        if len(args) == 1:
            limit = self.visit_expr(args[0])
            self.asm.ldi(iter_reg, 0)
        elif len(args) >= 2:
            # range(start, stop, step)
            start_reg = self.visit_expr(args[0])
            self.asm.mov(iter_reg, start_reg)
            self.free_if_temp(start_reg)
            limit = self.visit_expr(args[1])
            if len(args) == 3:
                step_reg = self.visit_expr(args[2])
                if isinstance(args[2], ast.Constant):
                    step = args[2].value
                else:
                    pass
        
        lbl_start = self.new_label("LOOP")
        lbl_end = self.new_label("END")
        
        self.asm.label(lbl_start)
        # Check cond: if iter >= limit, jump end
        # We only have BEQ/BNE. So we need SUB. 
        # If (limit - iter) <= 0 ? 
        # Vibe ASM is limited. Let's assume `i < N` loop.
        # Loop until i == N.
        self.asm.beq(iter_reg, limit, lbl_end)
        
        for s in node.body: self.visit(s)
        
        # Increment
        tmp = self.allocator.alloc()
        self.asm.ldi(tmp, step)
        self.asm.add(iter_reg, iter_reg, tmp)
        self.allocator.free(tmp)
        
        self.asm.jump(lbl_start)
        self.asm.label(lbl_end)
        # self.asm.join() # Remove JOIN to prevent incorrect merging with outer scope
        
        self.allocator.free_var(iter_var)
        self.free_if_temp(limit)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call):
            pass

    def visit_Assign(self, node):
        # Handle array assignment: c[i] = val
        target = node.targets[0]
        if isinstance(target, ast.Subscript):
            # Store
            arr_ptr = self.allocator.get(target.value.id) # Base addr
            idx_reg = self.visit_expr(target.slice) # Index
            
            val_reg = self.visit_expr(node.value) # Value to store
            
            # Address = ptr + idx * 4
            offset = self.allocator.alloc()
            self.asm.ldi(offset, 2)
            self.asm.sll(offset, idx_reg, offset) # idx * 4
            self.asm.add(offset, arr_ptr, offset) # addr
            
            self.asm.stw(offset, val_reg)
            
            self.allocator.free(offset)
            self.free_if_temp(idx_reg)
            self.free_if_temp(val_reg)
        else:
            # Variable assign
            var_name = target.id
            reg_val = self.visit_expr(node.value)
            
            existing = self.allocator.get(var_name)
            if existing:
                self.asm.mov(existing, reg_val)
                self.allocator.free(reg_val)
            else:
                self.allocator.var_map[var_name] = reg_val

    def visit_Subscript(self, node):
        # Load: x = a[i]
        # context Load
        arr_ptr = self.allocator.get(node.value.id)
        idx_reg = self.visit_expr(node.slice)
        
        offset = self.allocator.alloc()
        self.asm.ldi(offset, 2)
        self.asm.sll(offset, idx_reg, offset) # idx * 4
        self.asm.add(offset, arr_ptr, offset)
        
        res = self.allocator.alloc()
        self.asm.ldw(res, offset)
        
        self.allocator.free(offset)
        self.free_if_temp(idx_reg)
        return res

# --- Runtime Helper ---
class CudaRuntime:
    class Dim3:
        def __init__(self): self.x=0; self.y=0; self.z=0
    
    def __init__(self):
        self.threadIdx = self.Dim3()
        self.blockIdx = self.Dim3()
        self.blockDim = self.Dim3()
        self.gridDim = self.Dim3()
        
    def grid(self, dim): return 0
    def csr_cycle(self): return 0
    def csr_instret(self): return 0
    def syncthreads(self): pass
    
    def jit(self, func=None, block_dim=32):
        if func is None:
            return lambda f: self._compile(f, block_dim)
        return self._compile(func, block_dim)

    def _compile(self, func, block_dim):
        def wrapper(*args):
            arg_names = inspect.signature(func).parameters.keys()
            arg_vals = dict(zip(arg_names, args))
            
            asm = Assembler(base_data_addr=1024)
            compiler = CudaCompiler(asm, arg_vals, block_dim)
            compiler.compile(func)
            print(f"CUDA Compilation (Block={block_dim}) Done. Insts: {len(asm.insts)}")
            return asm
        return wrapper

_runtime = CudaRuntime()
jit = _runtime.jit
threadIdx = _runtime.threadIdx
blockIdx = _runtime.blockIdx
blockDim = _runtime.blockDim
gridDim = _runtime.gridDim
grid = _runtime.grid
csr_cycle = _runtime.csr_cycle
csr_instret = _runtime.csr_instret
syncthreads = _runtime.syncthreads
