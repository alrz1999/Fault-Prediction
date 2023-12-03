import re

import javalang


class TokenExtractor:
    def extract_tokens(self, input_text):
        pass


class CustomTokenExtractor(TokenExtractor):
    def __init__(self, to_lowercase=False, max_seq_len=None):
        self.to_lowercase = to_lowercase
        self.max_seq_len = max_seq_len

    def extract_tokens(self, input_text):
        input_text = re.sub('\\s+', ' ', input_text)
        input_text = self.preprocess_code_line(input_text)
        if self.to_lowercase:
            input_text = input_text.lower()

        tokens = input_text.strip().split()

        if self.max_seq_len is not None:
            tokens_count = len(tokens)

            tokens = tokens[:self.max_seq_len]

            if tokens_count < self.max_seq_len:
                tokens = tokens + ['<pad>'] * (self.max_seq_len - tokens_count)

        return tokens

    @classmethod
    def preprocess_code_line(cls, code_line):
        """
            input
                code_line (string)
        """
        CHAR_TO_REMOVE = ['+', '-', '*', '/', '=', '++', '--', '\\', '<str>', '<char>', '|', '&', '!']

        code_line = re.sub(r"\'\'", "\'", code_line)
        code_line = re.sub(r"\".*?\"", "<str>", code_line)
        code_line = re.sub(r"\'.*?\'", "<char>", code_line)
        code_line = re.sub(r"\b\d+[xX]?\d*[abcdexkDFLfl]*\d*\b", '<num>', code_line)
        code_line = re.sub(r"\\[.*?]", '', code_line)
        code_line = re.sub(r"[.|,:;{}()]", ' ', code_line)

        for char in CHAR_TO_REMOVE:
            code_line = code_line.replace(char, ' ')

        code_line = code_line.strip()

        return code_line


class RawTextTokenExtractor(TokenExtractor):
    def extract_tokens(self, input_text):
        # Split the input text into tokens using whitespace as the delimiter
        tokens = input_text.split()
        return tokens


class CFGTokenExtractor(TokenExtractor):
    def __init__(self):
        # Define regular expressions for different Java token types
        self.keywords = r'abstract|assert|boolean|break|byte|case|catch|char|class|const|continue|default|do|double|else|enum|extends|final|finally|float|for|if|implements|import|instanceof|int|interface|long|native|new|null|package|private|protected|public|return|short|static|strictfp|super|switch|synchronized|this|throw|throws|transient|try|void|volatile|while|true|false'
        self.operators = r'[=+\-*/<>!&|%^~]'
        self.punctuation = r'[(),.:;]'
        self.string_literal = r'"[^"]*"'
        self.comment = r'//.*|/\*[\s\S]*?\*/'
        self.identifier = r'[a-zA-Z_]\w*'

        # Combine all the regular expressions into one
        self.token_regex = '|'.join(
            [self.keywords, self.operators, self.punctuation, self.string_literal, self.comment, self.identifier])

    def extract_tokens(self, input_text):
        tokens = []
        for match in re.finditer(self.token_regex, input_text):
            token = match.group(0).strip()
            if token:
                tokens.append(token)
        return tokens


class ASTTokenizer(TokenExtractor):
    def __init__(self, cross_project=False):
        self.cross_project = cross_project

    def extract_tokens(self, input_text):
        # TODO line by line tokenization can be supported
        # TODO masking and using placeholder instead of some types like Identifiers
        try:
            tokens = []
            for token in javalang.tokenizer.tokenize(input_text, ignore_errors=True):
                if isinstance(token, javalang.tokenizer.Separator):
                    continue

                if isinstance(token, ASTTokenizer.get_number_token_types()):
                    tokens.append("<num>")
                elif isinstance(token, javalang.tokenizer.String):
                    tokens.append("<str>")
                elif isinstance(token, javalang.tokenizer.Character):
                    tokens.append("<char>")
                elif isinstance(token, javalang.tokenizer.Boolean):
                    tokens.append("<bool>")
                elif isinstance(token, javalang.tokenizer.Operator):
                    tokens.append("<op>")
                elif self.cross_project and isinstance(token, javalang.tokenizer.Identifier):
                    tokens.append("<identifier>")
                else:
                    tokens.append(token.value)

                if isinstance(token, javalang.tokenizer.Modifier):
                    pass
                if isinstance(token, javalang.tokenizer.BasicType):
                    pass
            return tokens
        except:
            return CustomTokenExtractor().extract_tokens(input_text)

    @staticmethod
    def get_number_token_types():
        return (
            javalang.tokenizer.Integer,
            javalang.tokenizer.DecimalInteger,
            javalang.tokenizer.BinaryInteger,
            javalang.tokenizer.OctalInteger,
            javalang.tokenizer.HexInteger,

            javalang.tokenizer.FloatingPoint,
            javalang.tokenizer.DecimalFloatingPoint,
            javalang.tokenizer.HexFloatingPoint,
        )


class ASTExtractor(TokenExtractor):
    def __init__(self, within_project=True):
        self.within_project = within_project

    method_invocations_and_class_instance_creation_nodes = (
        javalang.parser.tree.MethodInvocation,
        javalang.parser.tree.SuperMethodInvocation,
        javalang.parser.tree.MemberReference,
        javalang.parser.tree.SuperMemberReference,
    )

    declaration_nodes = (
        javalang.parser.tree.PackageDeclaration,
        javalang.parser.tree.InterfaceDeclaration,
        javalang.parser.tree.ClassDeclaration,
        javalang.parser.tree.MethodDeclaration,
        javalang.parser.tree.ConstructorDeclaration,
        javalang.parser.tree.VariableDeclarator,
        javalang.parser.tree.CatchClauseParameter,
        javalang.parser.tree.FormalParameter,
        javalang.parser.tree.TryResource,
        javalang.parser.tree.ReferenceType,
        javalang.parser.tree.BasicType,
    )

    control_flow_nodes = (
        javalang.parser.tree.IfStatement,
        javalang.parser.tree.WhileStatement,
        javalang.parser.tree.DoStatement,
        javalang.parser.tree.ForStatement,
        javalang.parser.tree.AssertStatement,
        javalang.parser.tree.BreakStatement,
        javalang.parser.tree.ContinueStatement,
        javalang.parser.tree.ReturnStatement,
        javalang.parser.tree.ThrowStatement,
        javalang.parser.tree.SynchronizedStatement,
        javalang.parser.tree.TryStatement,
        javalang.parser.tree.SwitchStatement,
        javalang.parser.tree.CatchClause,
        javalang.parser.tree.BlockStatement,
        javalang.parser.tree.StatementExpression,
        javalang.parser.tree.ForControl,
        javalang.parser.tree.SwitchStatementCase,
        javalang.parser.tree.EnhancedForControl,
    )

    desired_nodes = (
        *method_invocations_and_class_instance_creation_nodes,
        *declaration_nodes,
        *control_flow_nodes
    )

    within_project_nodes = (
        javalang.parser.tree.ClassDeclaration,
        javalang.parser.tree.MethodDeclaration,
        javalang.parser.tree.FormalParameter,
        javalang.parser.tree.MethodInvocation,
        javalang.parser.tree.CatchClauseParameter,
        javalang.parser.tree.ConstructorDeclaration,
    )

    def extract_tokens(self, input_text):
        tree = javalang.parse.parse(input_text)
        tokens = []
        for path, node in tree:
            if isinstance(node, ASTExtractor.desired_nodes):
                if self.within_project and isinstance(node, ASTExtractor.within_project_nodes):
                    if isinstance(node, javalang.parser.tree.MethodInvocation):
                        tokens.append(f'{node.qualifier}()')
                    else:
                        tokens.append(node.name)
                else:
                    tokens.append(node.__class__.__name__)

        return tokens


def test():
    input_text = """
    // Salam
    public class Example {
        int x = 10;
        public bool myMethod(Mm in1) {
            String message = "Hello, World!";
            if (a == 1){
            System.out.println(message);
            }
        }
    }
    """
    tree = javalang.parse.parse(input_text)
    # check_tree(tree)
    print(ASTExtractor().extract_tokens(input_text))


def check_tree(tree):
    depth = 1
    final_str = ""
    for path, node in tree:
        print("----------------------------------------------------")

        if len(path) < depth:
            final_str += "\n"
        depth = len(path)
        if hasattr(node, 'modifiers') and len(node.modifiers) > 0:
            final_str += f" {list(node.modifiers)[0]}"
        if hasattr(node, 'return_type'):
            if node.return_type is None:
                final_str += " void "
        if isinstance(node, javalang.parser.tree.ClassDeclaration):
            final_str += " class "
        if hasattr(node, 'name'):
            final_str += f" {node.name} "
        if hasattr(node, 'value'):
            final_str += f" {node.value}"

        print(type(path), len(path))
        for x in path:
            if isinstance(x, list):
                print([type(y) for y in x])
            else:
                print(type(x))
        print(node.__class__.__name__)
        print(node)
        print(node.children)
        print(node.position)
        print("----------------------------------------------------")
    print(final_str)


if __name__ == "__main__":
    test()
    # TODO
    # return file level tokens
    # return class level tokens
    # return method level tokens
    # return line level tokens
