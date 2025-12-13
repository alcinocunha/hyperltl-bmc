"""
(Complete) Bounded Model Checker for HyperLTL specifications over SMV models
"""


from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto
from z3 import *
import itertools
from math import lcm
import sys

"""
SMV Parser (by Claude Opus 4.5 adapted by Alcino Cunha)
Supports:  MODULE, VAR, FROZENVAR, INIT, TRANS, INVAR, and HLTLSPEC sections.
"""

# ============================================================================
# AST Node Definitions
# ============================================================================

class NodeType(Enum):
    # Literals
    INTEGER = auto()
    BOOLEAN = auto()
    IDENTIFIER = auto()
    
    # Expressions
    UNARY_OP = auto()
    BINARY_OP = auto()
    NEXT_EXPR = auto()
    RANGE_EXPR = auto()
    QUANTIFIED_EXPR = auto()
    
    # Types
    BOOLEAN_TYPE = auto()
    RANGE_TYPE = auto()
    
    # Declarations
    VAR_DECL = auto()
    
    # Module
    MODULE = auto()


@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    node_type: NodeType


@dataclass
class IntegerLiteral(ASTNode):
    value: int
    
    def __init__(self, value: int):
        super().__init__(NodeType.INTEGER)
        self.value = value
    
    def __repr__(self):
        return f"{self.value}"


@dataclass
class BooleanLiteral(ASTNode):
    value: bool
    
    def __init__(self, value: bool):
        super().__init__(NodeType.BOOLEAN)
        self.value = value
    
    def __repr__(self):
        return "TRUE" if self.value else "FALSE"


@dataclass
class Identifier(ASTNode):
    name: str
    trace: Optional[str] = None  # For trace quantification
    
    def __init__(self, name: str, trace: Optional[str] = None):
        super().__init__(NodeType.IDENTIFIER)
        self.name = name
        self.trace = trace
    
    def __repr__(self):
        if self.trace:
            return f"{self.name}[{self.trace}]"
        return self.name


@dataclass
class UnaryOp(ASTNode):
    operator: str
    operand: ASTNode
    
    def __init__(self, operator: str, operand: ASTNode):
        super().__init__(NodeType.UNARY_OP)
        self.operator = operator
        self.operand = operand
    
    def __repr__(self):
        return f"({self.operator} {self.operand})"


@dataclass
class BinaryOp(ASTNode):
    operator: str
    left: ASTNode
    right: ASTNode
    
    def __init__(self, operator: str, left: ASTNode, right: ASTNode):
        super().__init__(NodeType.BINARY_OP)
        self.operator = operator
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"({self.left} {self.operator} {self.right})"


@dataclass
class NextExpr(ASTNode):
    expr: ASTNode
    
    def __init__(self, expr:  ASTNode):
        super().__init__(NodeType.NEXT_EXPR)
        self.expr = expr
    
    def __repr__(self):
        return f"next({self.expr})"

# Type nodes
@dataclass
class BooleanType(ASTNode):
    def __init__(self):
        super().__init__(NodeType.BOOLEAN_TYPE)
    
    def __repr__(self):
        return "boolean"


@dataclass
class RangeType(ASTNode):
    lower: int
    upper: int
    
    def __init__(self, lower: int, upper: int):
        super().__init__(NodeType.RANGE_TYPE)
        self.lower = lower
        self.upper = upper
    
    def __repr__(self):
        return f"{self.lower}..{self. upper}"

@dataclass
class VarDecl(ASTNode):
    name: str
    var_type: ASTNode
    
    def __init__(self, name: str, var_type: ASTNode):
        super().__init__(NodeType.VAR_DECL)
        self.name = name
        self.var_type = var_type
    
    def __repr__(self):
        return f"{self.name} : {self. var_type}"


@dataclass
class QuantifiedExpr(ASTNode):
    quantifiers: list  # sequence of 'forall' or 'exists'
    vars: list
    expr: ASTNode
    
    def __init__(self, quantifiers: list, vars: list, expr: ASTNode):
        super().__init__(NodeType.QUANTIFIED_EXPR)
        self.quantifiers = quantifiers
        self.vars = vars
        self.expr = expr
    
    def __repr__(self):
        return ".".join([f"{q} {v}" for q, v in zip(self.quantifiers, self.vars)] + [f"({self.expr})"])

@dataclass
class Module(ASTNode):
    name: str
    var_decls: list = field(default_factory=list)
    frozenvar_decls: list = field(default_factory=list)
    init_expr: Optional[ASTNode] = None
    trans_expr: Optional[ASTNode] = None
    invar_expr: Optional[ASTNode] = None
    hltl_expr: ASTNode = None
    define_decls: list = field(default_factory=list)
    
    def __init__(self, name:  str):
        super().__init__(NodeType.MODULE)
        self.name = name
        self.var_decls = []
        self.frozenvar_decls = []
        self.init_expr = None
        self.trans_expr = None
        self.invar_expr = None
        self.hltl_expr = None
        self.define_decls = []
    
    def __repr__(self):
        parts = [f"MODULE {self.name}"]
        if self.frozenvar_decls:
            parts.append("FROZENVAR")
            for decl in self.frozenvar_decls:
                parts. append(f"    {decl};")
        if self.var_decls:
            parts. append("VAR")
            for decl in self.var_decls:
                parts.append(f"    {decl};")
        if self.init_expr:
            parts.append(f"INIT\n    {self.init_expr}")
        if self.trans_expr:
            parts.append(f"TRANS\n    {self. trans_expr}")
        if self.invar_expr:
            parts.append(f"INVAR\n    {self.invar_expr}")
        if self.hltl_expr:
            parts.append(f"HLTLSPEC\n    {self.hltl_expr}")
        return "\n".join(parts)

# ============================================================================
# Tokenizer
# ============================================================================

class TokenType(Enum):
    # Keywords
    MODULE = auto()
    VAR = auto()
    FROZENVAR = auto()
    INIT = auto()
    TRANS = auto()
    INVAR = auto()
    NEXT = auto()
    TRUE = auto()
    FALSE = auto()
    BOOLEAN = auto()
    HLTLSPEC = auto()
    
    # Literals and identifiers
    INTEGER = auto()
    IDENTIFIER = auto()
    
    # Operators
    COLON = auto()       # :
    SEMICOLON = auto()   # ;
    COMMA = auto()       # ,
    DOT = auto()         # .
    DOTDOT = auto()      # ..
    LPAREN = auto()      # (
    RPAREN = auto()      # )
    LBRACE = auto()      # {
    RBRACE = auto()      # }
    LBRACKET = auto()    # [
    RBRACKET = auto()    # ]
    
    # Logical operators
    AND = auto()         # &
    OR = auto()          # |
    NOT = auto()         # !
    IMPLIES = auto()     # ->
    IFF = auto()         # <->
    
    # Comparison operators
    EQ = auto()          # =
    NEQ = auto()         # !=
    LT = auto()          # <
    LE = auto()          # <=
    GT = auto()          # >
    GE = auto()          # >=
    
    # Arithmetic operators
    PLUS = auto()        # +
    MINUS = auto()       # -
    TIMES = auto()       # *
    DIVIDE = auto()      # /
    MOD = auto()         # mod
    
    # Temporal operators
    ALWAYS = auto()      # G
    EVENTUALLY = auto()  # F

    # Trace quantifiers
    FORALL = auto()      # forall
    EXISTS = auto()      # exists

    # Special
    EOF = auto()
    NEWLINE = auto()


@dataclass
class Token:
    type: TokenType
    value:  any
    line: int
    column: int
    
    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, line={self.line}, col={self.column})"


class Tokenizer:
    KEYWORDS = {
        'MODULE':  TokenType.MODULE,
        'VAR': TokenType.VAR,
        'FROZENVAR': TokenType.FROZENVAR,
        'INIT': TokenType.INIT,
        'TRANS': TokenType.TRANS,
        'INVAR':  TokenType.INVAR,
        'HLTLSPEC': TokenType.HLTLSPEC,
        'next': TokenType.NEXT,
        'TRUE': TokenType.TRUE,
        'FALSE': TokenType.FALSE,
        'boolean': TokenType.BOOLEAN,
        'mod': TokenType.MOD,
        'forall': TokenType.FORALL,
        'exists': TokenType.EXISTS,
        'G': TokenType.ALWAYS,
        'F': TokenType.EVENTUALLY,
    }
    
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens = []
    
    def error(self, message: str):
        raise SyntaxError(f"Tokenizer error at line {self.line}, column {self.column}: {message}")
    
    def peek(self, offset: int = 0) -> Optional[str]:
        pos = self.pos + offset
        if pos < len(self.text):
            return self.text[pos]
        return None
    
    def advance(self) -> Optional[str]:
        if self.pos < len(self. text):
            char = self.text[self.pos]
            self.pos += 1
            if char == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            return char
        return None
    
    def skip_whitespace_and_comments(self):
        while self.pos < len(self.text):
            char = self.peek()
            
            # Skip whitespace
            if char in ' \t\r\n': 
                self.advance()
            # Skip single-line comments
            elif char == '-' and self.peek(1) == '-':
                while self.peek() and self.peek() != '\n':
                    self.advance()
            # Skip multi-line comments
            elif char == '/' and self.peek(1) == '*':
                self.advance()  # /
                self.advance()  # *
                while self.pos < len(self.text):
                    if self.peek() == '*' and self.peek(1) == '/':
                        self.advance()  # *
                        self. advance()  # /
                        break
                    self.advance()
            else:
                break
    
    def read_number(self) -> Token:
        start_line, start_col = self.line, self.column
        num_str = ''
        
        # Handle negative numbers
        if self. peek() == '-':
            num_str += self.advance()
        
        while self.peek() and self.peek().isdigit():
            num_str += self.advance()
        
        return Token(TokenType.INTEGER, int(num_str), start_line, start_col)
    
    def read_identifier(self) -> Token:
        start_line, start_col = self.line, self.column
        ident = ''
        
        while self.peek() and (self.peek().isalnum() or self.peek() in '_$#'):
            ident += self. advance()
        
        # Check if it's a keyword
        if ident in self.KEYWORDS:
            return Token(self. KEYWORDS[ident], ident, start_line, start_col)
        
        return Token(TokenType.IDENTIFIER, ident, start_line, start_col)
    
    def tokenize(self) -> list:
        self.tokens = []
        
        while self.pos < len(self.text):
            self.skip_whitespace_and_comments()
            
            if self.pos >= len(self.text):
                break
            
            char = self.peek()
            start_line, start_col = self.line, self.column
            
            # Numbers
            if char.isdigit():
                self.tokens.append(self.read_number())
            
            # Identifiers and keywords
            elif char.isalpha() or char == '_':
                self.tokens.append(self.read_identifier())
            
            # Two-character operators
            elif char == ':' and self.peek(1) == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.ASSIGN, ':=', start_line, start_col))
            
            elif char == '.' and self.peek(1) == '.':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType. DOTDOT, '..', start_line, start_col))
            
            elif char == '-' and self.peek(1) == '>':
                self. advance()
                self.advance()
                self.tokens.append(Token(TokenType.IMPLIES, '->', start_line, start_col))
            
            elif char == '<' and self.peek(1) == '-' and self.peek(2) == '>':
                self.advance()
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.IFF, '<->', start_line, start_col))
            
            elif char == '!' and self.peek(1) == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.NEQ, '!=', start_line, start_col))
            
            elif char == '<' and self.peek(1) == '=':
                self. advance()
                self.advance()
                self.tokens.append(Token(TokenType.LE, '<=', start_line, start_col))
            
            elif char == '>' and self.peek(1) == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType. GE, '>=', start_line, start_col))
            
            # Single-character operators
            elif char == ':':
                self.advance()
                self.tokens.append(Token(TokenType.COLON, ':', start_line, start_col))
            
            elif char == ';':
                self.advance()
                self. tokens.append(Token(TokenType.SEMICOLON, ';', start_line, start_col))
            
            elif char == ',':
                self.advance()
                self.tokens.append(Token(TokenType.COMMA, ',', start_line, start_col))
            
            elif char == '.':
                self.advance()
                self.tokens.append(Token(TokenType.DOT, '.', start_line, start_col))
            
            elif char == '(':
                self. advance()
                self.tokens. append(Token(TokenType. LPAREN, '(', start_line, start_col))
            
            elif char == ')':
                self.advance()
                self.tokens.append(Token(TokenType.RPAREN, ')', start_line, start_col))
            
            elif char == '{':
                self.advance()
                self.tokens.append(Token(TokenType.LBRACE, '{', start_line, start_col))
            
            elif char == '}':
                self.advance()
                self.tokens.append(Token(TokenType. RBRACE, '}', start_line, start_col))
            
            elif char == '[': 
                self.advance()
                self.tokens.append(Token(TokenType.LBRACKET, '[', start_line, start_col))
            
            elif char == ']':
                self.advance()
                self.tokens.append(Token(TokenType. RBRACKET, ']', start_line, start_col))
            
            elif char == '&':
                self.advance()
                self.tokens.append(Token(TokenType.AND, '&', start_line, start_col))
            
            elif char == '|':
                self.advance()
                self.tokens.append(Token(TokenType.OR, '|', start_line, start_col))
            
            elif char == '!':
                self.advance()
                self.tokens.append(Token(TokenType. NOT, '!', start_line, start_col))
            
            elif char == '=': 
                self.advance()
                self.tokens.append(Token(TokenType.EQ, '=', start_line, start_col))
            
            elif char == '<':
                self.advance()
                self.tokens.append(Token(TokenType.LT, '<', start_line, start_col))
            
            elif char == '>':
                self.advance()
                self.tokens.append(Token(TokenType.GT, '>', start_line, start_col))
            
            elif char == '+':
                self.advance()
                self.tokens.append(Token(TokenType.PLUS, '+', start_line, start_col))
            
            elif char == '-':
                self. advance()
                self.tokens. append(Token(TokenType. MINUS, '-', start_line, start_col))
            
            elif char == '*':
                self.advance()
                self.tokens.append(Token(TokenType. TIMES, '*', start_line, start_col))
            
            elif char == '/':
                self.advance()
                self. tokens.append(Token(TokenType.DIVIDE, '/', start_line, start_col))
            
            else:
                self.error(f"Unexpected character:  {char !r}")
        
        self.tokens.append(Token(TokenType.EOF, None, self. line, self.column))
        return self.tokens


# ============================================================================
# Parser
# ============================================================================

class Parser: 
    # Section keywords that can start a new section
    SECTION_KEYWORDS = {
        TokenType.VAR, TokenType.FROZENVAR, TokenType.INIT,
        TokenType.TRANS, TokenType.INVAR, TokenType.HLTLSPEC, TokenType.MODULE
    }
    
    def __init__(self, tokens: list):
        self.tokens = tokens
        self.pos = 0
    
    def error(self, message: str):
        token = self.current()
        raise SyntaxError(f"Parse error at line {token.line}, column {token.column}: {message}")
    
    def current(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # Return EOF
    
    def peek(self, offset: int = 0) -> Token:
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return self.tokens[-1]
    
    def advance(self) -> Token:
        token = self.current()
        if self.pos < len(self.tokens) - 1:
            self. pos += 1
        return token
    
    def expect(self, token_type: TokenType, message: str = None) -> Token:
        if self.current().type != token_type:
            msg = message or f"Expected {token_type}, got {self.current().type}"
            self.error(msg)
        return self. advance()
    
    def match(self, *token_types:  TokenType) -> bool:
        return self.current().type in token_types
    
    def is_section_keyword(self) -> bool:
        return self.current().type in self.SECTION_KEYWORDS
    
    # ========================================================================
    # Expression Parsing (Precedence Climbing)
    # ========================================================================
    
    def parse_expression(self) -> ASTNode:
        return self.parse_iff()
    
    def parse_iff(self) -> ASTNode:
        left = self.parse_implies()
        while self.match(TokenType.IFF):
            op = self.advance().value
            right = self.parse_implies()
            left = BinaryOp(op, left, right)
        return left
    
    def parse_implies(self) -> ASTNode:
        left = self.parse_or()
        while self.match(TokenType.IMPLIES):
            op = self.advance().value
            right = self.parse_or()
            left = BinaryOp(op, left, right)
        return left
    
    def parse_or(self) -> ASTNode:
        left = self.parse_and()
        while self.match(TokenType. OR):
            op = self. advance().value
            right = self.parse_and()
            left = BinaryOp(op, left, right)
        return left
    
    def parse_and(self) -> ASTNode:
        left = self.parse_comparison()
        while self.match(TokenType.AND):
            op = self.advance().value
            right = self.parse_comparison()
            left = BinaryOp(op, left, right)
        return left
    
    def parse_comparison(self) -> ASTNode:
        left = self.parse_additive()
        while self.match(TokenType.EQ, TokenType.NEQ, TokenType. LT, 
                         TokenType.LE, TokenType.GT, TokenType.GE):
            op = self.advance().value
            right = self.parse_additive()
            left = BinaryOp(op, left, right)
        return left
    
    def parse_additive(self) -> ASTNode:
        left = self.parse_multiplicative()
        while self.match(TokenType.PLUS, TokenType. MINUS):
            op = self. advance().value
            right = self.parse_multiplicative()
            left = BinaryOp(op, left, right)
        return left
    
    def parse_multiplicative(self) -> ASTNode:
        left = self. parse_unary()
        while self.match(TokenType. TIMES, TokenType.DIVIDE, TokenType.MOD):
            op = self.advance().value
            right = self.parse_unary()
            left = BinaryOp(op, left, right)
        return left
    
    def parse_unary(self) -> ASTNode:
        if self.match(TokenType.NOT):
            op = self.advance().value
            operand = self.parse_unary()
            return UnaryOp(op, operand)
        if self.match(TokenType.MINUS):
            op = self. advance().value
            operand = self.parse_unary()
            return UnaryOp(op, operand)
        if self.match(TokenType.ALWAYS):
            op = self.advance().value
            operand = self.parse_unary()
            return UnaryOp(op, operand)
        if self.match(TokenType.EVENTUALLY):
            op = self.advance().value
            operand = self.parse_unary()
            return UnaryOp(op, operand)
        return self.parse_primary()
    
    def parse_primary(self) -> ASTNode:
        token = self.current()
        
        # Integer literal
        if self.match(TokenType.INTEGER):
            self.advance()
            return IntegerLiteral(token.value)
        
        # Boolean literals
        if self.match(TokenType.TRUE):
            self.advance()
            return BooleanLiteral(True)
        
        if self.match(TokenType.FALSE):
            self.advance()
            return BooleanLiteral(False)
        
        # next(expr)
        if self.match(TokenType.NEXT):
            self.advance()
            self.expect(TokenType.LPAREN)
            expr = self.parse_identifier()
            self.expect(TokenType.RPAREN)
            return NextExpr(expr)
                
        # Parenthesized expression
        if self.match(TokenType.LPAREN):
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
        
        # Identifier
        if self.match(TokenType.IDENTIFIER):
            return self.parse_identifier()
        
        self.error(f"Unexpected token in expression: {token}")
    
    def parse_identifier(self) -> ASTNode:
        name = self.expect(TokenType.IDENTIFIER).value
        trace = None
        # Check for trace quantification
        if self.match(TokenType.LBRACKET):
            self.advance()  # [
            trace = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.RBRACKET)  # ]
            result = Identifier(name, trace)
        else:
            result = Identifier(name)        
        return result
    
    def parse_quantified_expression(self) -> ASTNode:
        quantifiers = []
        vars = []
        
        while self.match(TokenType.FORALL, TokenType.EXISTS):
            quant_token = self.advance()
            quantifiers.append(quant_token.value)
            var_name = self.expect(TokenType.IDENTIFIER).value
            vars.append(var_name)
            self.expect(TokenType.DOT)
        
        expr = self.parse_expression()
        return QuantifiedExpr(quantifiers, vars, expr)
    
    # ========================================================================
    # Type Parsing
    # ========================================================================
    
    def parse_type(self) -> ASTNode:
        # boolean type
        if self.match(TokenType. BOOLEAN):
            self.advance()
            return BooleanType()
        
        # Range type:  lower..upper
        if self.match(TokenType.INTEGER):
            lower = self.advance().value
            self.expect(TokenType. DOTDOT)
            upper = self.expect(TokenType.INTEGER).value
            return RangeType(lower, upper)
        
        self.error(f"Expected type, got {self.current()}")
    
    # ========================================================================
    # Section Parsing
    # ========================================================================
    
    def parse_var_section(self, is_frozen: bool = False) -> list:
        """Parse VAR or FROZENVAR section."""
        if is_frozen:
            self.expect(TokenType.FROZENVAR)
        else:
            self.expect(TokenType.VAR)
        
        decls = []
        
        while self.match(TokenType.IDENTIFIER):
            name = self.advance().value
            self.expect(TokenType.COLON)
            var_type = self.parse_type()
            self.expect(TokenType.SEMICOLON)
            decls.append(VarDecl(name, var_type))
        
        return decls
    
    def parse_init_section(self) -> ASTNode:
        """Parse INIT section."""
        self.expect(TokenType.INIT)
        expr = self.parse_expression()
        return expr
    
    def parse_trans_section(self) -> ASTNode:
        """Parse TRANS section."""
        self.expect(TokenType.TRANS)
        expr = self.parse_expression()
        return expr
    
    def parse_invar_section(self) -> ASTNode:
        """Parse INVAR section."""
        self.expect(TokenType.INVAR)
        expr = self.parse_expression()
        return expr
    
    def parse_hltlspec_section(self) -> ASTNode:
        """Parse HLTLSPEC section."""
        self.expect(TokenType.HLTLSPEC)
        qexpr = self.parse_quantified_expression()
        return qexpr

    
    # ========================================================================
    # Module Parsing
    # ========================================================================
    
    def parse_module(self) -> Module:
        """Parse a MODULE declaration."""
        self.expect(TokenType.MODULE)
        
        name = self.expect(TokenType. IDENTIFIER).value
        module = Module(name)
        
        # Parse sections
        while not self.match(TokenType.EOF, TokenType.MODULE):
            if self.match(TokenType.VAR):
                module.var_decls.extend(self.parse_var_section(is_frozen=False))
            elif self.match(TokenType.FROZENVAR):
                module.frozenvar_decls.extend(self.parse_var_section(is_frozen=True))
            elif self.match(TokenType. INIT):
                module.init_expr = self.parse_init_section()
            elif self.match(TokenType.TRANS):
                module.trans_expr = self. parse_trans_section()
            elif self.match(TokenType. INVAR):
                module. invar_expr = self.parse_invar_section()
            elif self.match(TokenType. HLTLSPEC):
                module.hltlspec_expr = self.parse_hltlspec_section()
            else:
                self.error(f"Unexpected token in module:  {self.current()}")
        
        return module

# ============================================================================
# Main Parser Function
# ============================================================================

def parse_smv(text: str) -> Module:
    """
    Parse SMV code and return a Module AST nodes.
    
    Args:
        text: The SMV source code as a string. 
    
    Returns:
        A Module object representing the parsed module.
    
    Raises:
        SyntaxError: If the input contains invalid syntax.
    """
    tokenizer = Tokenizer(text)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    return parser.parse_module()

# ===========================================================================
# Translate SMV expressions to Z3 (by Alcino Cunha and CoPilot)
# ===========================================================================

def smv_temporal_expr_to_z3(K, i, expr: ASTNode, state: dict, loops: dict) -> ExprRef:
    if isinstance(expr, IntegerLiteral):
        return IntVal(expr.value)
    elif isinstance(expr, BooleanLiteral):
        return BoolVal(expr.value)
    elif isinstance(expr, Identifier):
        if expr.trace is not None:
            loop = loops[expr.trace]
            lasso = K - loop
            if i < loop:
                index = i
            else:
                index = loop + ((i - loop) % lasso)
            return state[expr.trace][index][expr.name]
        else:
            raise NotImplementedError("Non projected identifiers not supported in temporal formulas.")
    elif isinstance(expr, UnaryOp):
        if expr.operator == '!':
            operand = smv_temporal_expr_to_z3(K, i, expr.operand, state, loops)
            return Not(operand)
        elif expr.operator == '-':
            operand = smv_temporal_expr_to_z3(K, i, expr.operand, state, loops)
            return (- operand)
        elif expr.operator == 'G':
            lasso = { t: K - loops[t] for t in loops }
            loop = max(loops.values())
            length = loop + lcm(*lasso.values())
            return And([smv_temporal_expr_to_z3(K, j, expr.operand, state, loops) for j in range(min(i,loop), length)])
        elif expr.operator == 'F':
            lasso = { t: K - loops[t] for t in loops }
            loop = max(loops.values())
            length = loop + lcm(*lasso.values())
            return Or([smv_temporal_expr_to_z3(K, j, expr.operand, state, loops) for j in range(min(i,loop), length)])     
    elif isinstance(expr, BinaryOp):
        left = smv_temporal_expr_to_z3(K, i, expr.left, state, loops)
        right = smv_temporal_expr_to_z3(K, i, expr.right, state, loops)
        if expr.operator == '&':
            return And(left, right)
        elif expr.operator == '|':
            return Or(left, right)
        elif expr.operator == '->':
            return Implies(left, right)
        elif expr.operator == '<->':
            return left == right
        elif expr.operator == '=':
            return left == right
        elif expr.operator == '!=':
            return left != right
        elif expr.operator == '<':
            return left < right
        elif expr.operator == '<=':
            return left <= right
        elif expr.operator == '>':
            return left > right
        elif expr.operator == '>=':
            return left >= right
        elif expr.operator == '+':
            return left + right
        elif expr.operator == '-':
            return left - right
        elif expr.operator == '*':
            return left * right
        elif expr.operator == '/':
            return left / right
        elif expr.operator == 'mod':
            return left % right
    elif isinstance(expr, NextExpr):
        raise NotImplementedError("Next not allowed in temporal formulas.")
    
    raise NotImplementedError(f"Translation for {expr} not implemented.")

def smv_expr_to_z3(expr: ASTNode, state: dict) -> ExprRef:
    if isinstance(expr, IntegerLiteral):
        return IntVal(expr.value)
    elif isinstance(expr, BooleanLiteral):
        return BoolVal(expr.value)
    elif isinstance(expr, Identifier):
        if expr.trace is not None:
            raise NotImplementedError("Projected identifiers not supported in state formulas.")
        return state[expr.name]
    elif isinstance(expr, UnaryOp):
        operand = smv_expr_to_z3(expr.operand, state)
        if expr.operator == '!':
            return Not(operand)
        elif expr.operator == '-':
            return (- operand)
        elif expr.operator == 'G':
            raise NotImplementedError("Temporal operator not allowed in state formulas.")
        elif expr.operator == 'F':
            raise NotImplementedError("Temporal operator not allowed in state formulas.")
    elif isinstance(expr, BinaryOp):
        left = smv_expr_to_z3(expr.left, state)
        right = smv_expr_to_z3(expr.right, state)
        if expr.operator == '&':
            return And(left, right)
        elif expr.operator == '|':
            return Or(left, right)
        elif expr.operator == '->':
            return Implies(left, right)
        elif expr.operator == '<->':
            return left == right
        elif expr.operator == '=':
            return left == right
        elif expr.operator == '!=':
            return left != right
        elif expr.operator == '<':
            return left < right
        elif expr.operator == '<=':
            return left <= right
        elif expr.operator == '>':
            return left > right
        elif expr.operator == '>=':
            return left >= right
        elif expr.operator == '+':
            return left + right
        elif expr.operator == '-':
            return left - right
        elif expr.operator == '*':
            return left * right
        elif expr.operator == '/':
            return left / right
        elif expr.operator == 'mod':
            return left % right
    elif isinstance(expr, NextExpr):
        raise NotImplementedError("Next not allowed in state formulas.")
    
    raise NotImplementedError(f"Translation for {expr} not implemented.")

def smv_next_expr_to_z3(expr: ASTNode, state1: dict, state2: dict) -> ExprRef:
    if isinstance(expr, IntegerLiteral):
        return IntVal(expr.value)
    elif isinstance(expr, BooleanLiteral):
        return BoolVal(expr.value)
    elif isinstance(expr, Identifier):
        return state1[expr.name]
    elif isinstance(expr, UnaryOp):
        operand = smv_next_expr_to_z3(expr.operand, state1, state2)
        if expr.operator == '!':
            return Not(operand)
        elif expr.operator == '-':
            return (- operand)
    elif isinstance(expr, BinaryOp):
        left = smv_next_expr_to_z3(expr.left, state1, state2)
        right = smv_next_expr_to_z3(expr.right, state1, state2)
        if expr.operator == '&':
            return And(left, right)
        elif expr.operator == '|':
            return Or(left, right)
        elif expr.operator == '->':
            return Implies(left, right)
        elif expr.operator == '<->':
            return left == right
        elif expr.operator == '=':
            return left == right
        elif expr.operator == '!=':
            return left != right
        elif expr.operator == '<':
            return left < right
        elif expr.operator == '<=':
            return left <= right
        elif expr.operator == '>':
            return left > right
        elif expr.operator == '>=':
            return left >= right
        elif expr.operator == '+':
            return left + right
        elif expr.operator == '-':
            return left - right
        elif expr.operator == '*':
            return left * right
        elif expr.operator == '/':
            return left / right
        elif expr.operator == 'mod':
            return left % right
    elif isinstance(expr, NextExpr):
        return state2[expr.expr.name]
    
    raise NotImplementedError(f"Translation for {expr} not implemented.")

# ============================================================================
# BMC procedure for HyperLTL (by Alcino Cunha and CoPilot)
# ============================================================================

if __name__ == "__main__":
    # check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python bmc.py <model_file.smv> <trace_length>")
        sys.exit(1)
    model_file = sys.argv[1]
    K = int(sys.argv[2])
    
    # Read model from file
    with open(model_file, "r") as f:

        example_model = f.read()

        try:
            module = parse_smv(example_model)
            
            def frozen_vars(name):
                decls = {}
                for decl in module.frozenvar_decls:
                    if decl.var_type.__class__ == RangeType:
                        decls[decl.name] = Int(f"{name}_{decl.name}")
                    elif decl.var_type.__class__ == BooleanType:
                        decls[decl.name] = Bool(f"{name}_{decl.name}")
                return decls
            
            def state_vars(name,i):
                decls = {}
                for decl in module.var_decls:
                    if decl.var_type.__class__ == RangeType:
                        decls[decl.name] = Int(f"{name}_{i}_{decl.name}")
                    elif decl.var_type.__class__ == BooleanType:
                        decls[decl.name] = Bool(f"{name}_{i}_{decl.name}")
                return decls
            
            def init(s):
                return smv_expr_to_z3(module.init_expr, s,)
            
            def trans(s1,s2):
                return smv_next_expr_to_z3(module.trans_expr, s1, s2)

            def invar(s):
                if module.invar_expr is None:
                    return BoolVal(True)
                return smv_expr_to_z3(module.invar_expr, s)

            def behavior(state,loop,init,trans,invar):
                constraints = [loop >= 0, loop < K, init(state[0])]
                for decl in module.frozenvar_decls:
                    if decl.var_type.__class__ == RangeType:
                        var = state[0][decl.name]
                        constraints.append(var >= decl.var_type.lower)
                        constraints.append(var <= decl.var_type.upper)
                for i in range(K-1):
                    constraints.append(trans(state[i], state[i+1]))
                for i in range(K):
                    constraints.append(Implies(loop == i,trans(state[K-1], state[i])))
                for i in range(K):
                    constraints.append(invar(state[i]))
                for i in range(K):
                    for decl in module.var_decls:
                        if decl.var_type.__class__ == RangeType:
                            var = state[i][decl.name]
                            constraints.append(var >= decl.var_type.lower)
                            constraints.append(var <= decl.var_type.upper)
                return And(constraints)
            
            def print_trace(model,state,loop,K):
                for i in range(K):
                    print(f"State {i}:")
                    for var in state[i]:
                        print(f"  {var} = {model[state[i][var]]}")
                l = model[loop].as_long()
                print(f"Loop back to state: {l}")

            s = Solver()

            N = len(module.hltlspec_expr.vars)

            first_exists = N
            state = {}
            loop = {}
            for i,t in enumerate(module.hltlspec_expr.vars):
                if module.hltlspec_expr.quantifiers[i] == 'exists':
                    first_exists = min(first_exists, i)
                frozen = frozen_vars(t)
                state[t] = [state_vars(t, j) | frozen for j in range(K)]
                loop[t] = Int(f"loop_{t}")

            for i in range(first_exists):
                s.add(behavior(state[module.hltlspec_expr.vars[i]], loop[module.hltlspec_expr.vars[i]], init, trans, invar))

            exprs = []
            for l in itertools.product(range(K), repeat=N):
                loops = {}
                for i,t in enumerate(module.hltlspec_expr.vars):
                    loops[t] = l[i]    
                lhs = And(loop[t] == loops[t] for t in module.hltlspec_expr.vars)
                rhs = Not(smv_temporal_expr_to_z3(K, 0, module.hltlspec_expr.expr, state, loops))
                exprs.append(Implies(lhs, rhs))
            expr = And(exprs)

            for i in range(N-1, first_exists-1, -1):
                if module.hltlspec_expr.quantifiers[i] == 'forall':
                    t = module.hltlspec_expr.vars[i]
                    form = And(behavior(state[t], loop[t], init, trans, invar), expr)
                    expr = Exists([loop[t]] + [l for j in range(K) for l in state[t][j].values()], form)
                else:
                    t = module.hltlspec_expr.vars[i]
                    form = Implies(behavior(state[t], loop[t], init, trans, invar), expr)
                    expr = ForAll([loop[t]] + [l for j in range(K) for l in state[t][j].values()], form)
            s.add(expr)
            r = s.check()
            if r == sat:
                print(r)
                print("Counterexample found:")
                model = s.model()
                for i in range(first_exists):
                    print(f"* Trace {module.hltlspec_expr.vars[i]} *")
                    print_trace(model,state[module.hltlspec_expr.vars[i]],loop[module.hltlspec_expr.vars[i]],K)
            else:
                print(r)
                print("No counterexample found.")

        except SyntaxError as e:
            print(f"Error: {e}")