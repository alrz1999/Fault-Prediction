# token_extraction.py
import javalang
import re


class TokenExtractor:
    def extract_tokens(self, input_text):
        pass


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


class ASTTokenExtractor(TokenExtractor):
    def extract_tokens(self, input_text):
        # Parse the input Java code
        tree = javalang.parse.parse(input_text)

        # Initialize an empty list to store the tokens
        tokens = []

        # Define a function to recursively traverse the AST and extract tokens
        def traverse(node):
            if isinstance(node, javalang.parser.tree.VariableDeclarator):
                pass
            elif isinstance(node, javalang.parser.tree.MethodDeclaration):
                pass
            elif isinstance(node, javalang.parser.tree.ClassDeclaration):
                pass
            elif isinstance(node, javalang.parser.tree.Literal):
                pass
            elif isinstance(node, javalang.parser.JavaToken):
                print(node)
            elif isinstance(node, javalang.tokenizer.Identifier):
                pass

            if isinstance(node, javalang.parser.tree.VariableDeclarator):
                # Extract tokens from the children of VariableDeclarator nodes
                for child in node.children:
                    if isinstance(child, javalang.tokenizer.Identifier):
                        tokens.append(child.value)
            elif isinstance(node, javalang.parser.tree.MethodDeclaration):
                # Extract tokens from the children of MethodDeclaration nodes
                for child in node.children:
                    if isinstance(child, javalang.tokenizer.Identifier):
                        tokens.append(child.value)
            elif isinstance(node, javalang.parser.tree.ClassDeclaration):
                # Extract tokens from the children of ClassDeclaration nodes
                for child in node.children:
                    if isinstance(child, javalang.tokenizer.Identifier):
                        tokens.append(child.value)

            # Check if the node has children before trying to access them
            if hasattr(node, 'children'):
                for child in node.children:
                    traverse(child)

        # Start the traversal from the root of the AST
        for path, node in tree:
            traverse(node)

        return tokens

    @staticmethod
    def get_default_desired_token_types():
        return (
            javalang.tokenizer.Modifier,
            javalang.tokenizer.Keyword,
            javalang.tokenizer.Identifier,
            # javalang.tokenizer.Separator,
            javalang.tokenizer.BasicType,
            javalang.tokenizer.DecimalInteger,
            javalang.tokenizer.Operator,
            javalang.tokenizer.String,
        )

    @staticmethod
    def tokenize(input_text, desired_token_types=None):
        # TODO line by line tokenization can be supported
        # TODO masking and using placeholder instead of some types like Identifiers
        tokens = []

        if desired_token_types is None:
            desired_token_types = ASTTokenExtractor.get_default_desired_token_types()

        for token in javalang.tokenizer.tokenize(input_text):
            if isinstance(token, desired_token_types):
                tokens.append(token.value)

        return tokens


def test():
    input_text = """
    public class Example {
        int x = 10;
        public void myMethod() {
            String message = "Hello, World!";
            System.out.println(message);
        }
    }
    """
    extractor = ASTTokenExtractor()
    tokens = extractor.tokenize(input_text)
    print(tokens)


if __name__ == "__main__":
    test()
