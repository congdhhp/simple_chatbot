"""Command-line interface for the chatbot."""

import click
import logging
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.prompt import Prompt
from rich.markdown import Markdown

from config_manager import ConfigManager
from model_manager import ModelManager
from conversation_manager import ConversationManager


class ChatbotCLI:
    """Main CLI class for the chatbot."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.console = Console()
        self.config_manager = None
        self.model_manager = None
        self.conversation_manager = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('chatbot.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize(self, config_path: str = "config/models.yaml"):
        """Initialize the chatbot components."""
        try:
            self.config_manager = ConfigManager(config_path)
            self.model_manager = ModelManager(self.config_manager)
            self.conversation_manager = ConversationManager(self.config_manager)
            return True
        except Exception as e:
            self.console.print(f"[red]Error initializing chatbot: {e}[/red]")
            return False
    
    def show_welcome(self):
        """Display welcome message."""
        welcome_text = Text()
        welcome_text.append("🤖 Simple CLI Chatbot\n", style="bold blue")
        welcome_text.append("Powered by Hugging Face Transformers\n\n", style="dim")
        welcome_text.append("Commands:\n", style="bold")
        welcome_text.append("  /help     - Show this help message\n", style="cyan")
        welcome_text.append("  /models   - List available models\n", style="cyan")
        welcome_text.append("  /switch   - Switch to a different model\n", style="cyan")
        welcome_text.append("  /info     - Show current model info\n", style="cyan")
        welcome_text.append("  /config   - Show/modify model configuration\n", style="cyan")
        welcome_text.append("  /clear    - Clear conversation history\n", style="cyan")
        welcome_text.append("  /save     - Save conversation\n", style="cyan")
        welcome_text.append("  /load     - Load conversation\n", style="cyan")
        welcome_text.append("  /list     - List saved conversations\n", style="cyan")
        welcome_text.append("  /quit     - Exit the chatbot\n", style="cyan")
        welcome_text.append("\nType your message and press Enter to chat!", style="green")
        
        panel = Panel(welcome_text, title="Welcome", border_style="blue")
        self.console.print(panel)
    
    def show_models(self):
        """Display available models."""
        models = self.config_manager.get_available_models()
        current_model = self.model_manager.current_model_name
        default_model = self.config_manager.get_default_model()
        
        table = Table(title="Available Models")
        table.add_column("Name", style="cyan")
        table.add_column("Display Name", style="green")
        table.add_column("Status", style="yellow")
        
        for name, display_name in models.items():
            status = ""
            if name == current_model:
                status = "🟢 Loaded"
            elif name == default_model:
                status = "⭐ Default"
            
            table.add_row(name, display_name, status)
        
        self.console.print(table)
    
    def show_model_info(self):
        """Display current model information."""
        info = self.model_manager.get_current_model_info()
        
        if info.get("status") == "No model loaded":
            self.console.print("[yellow]No model currently loaded[/yellow]")
            return
        
        info_text = Text()
        info_text.append(f"Model: {info['display_name']}\n", style="bold green")
        info_text.append(f"ID: {info['model_id']}\n", style="dim")
        info_text.append(f"Device: {info['device']}\n", style="cyan")
        info_text.append(f"Description: {info.get('description', 'N/A')}\n", style="white")
        
        panel = Panel(info_text, title="Current Model Info", border_style="green")
        self.console.print(panel)
    
    def switch_model(self):
        """Switch to a different model."""
        models = self.config_manager.get_available_models()

        self.console.print("\n[bold]Available models:[/bold]")
        for i, (name, display_name) in enumerate(models.items(), 1):
            self.console.print(f"  {i}. {display_name} ({name})")

        choice = Prompt.ask(
            "\nEnter model number or name",
            choices=[str(i) for i in range(1, len(models) + 1)] + list(models.keys()),
            default="1"
        )

        # Convert number to model name if needed
        if choice.isdigit():
            model_name = list(models.keys())[int(choice) - 1]
        else:
            model_name = choice

        self.console.print(f"\n[yellow]Loading model: {models[model_name]}...[/yellow]")

        # Unload current model first
        if self.model_manager.current_model is not None:
            self.model_manager.unload_model()

        # Load new model
        if self.model_manager.load_model(model_name):
            self.console.print(f"[green]✓ Model loaded successfully![/green]")
        else:
            self.console.print(f"[red]✗ Failed to load model[/red]")

    def show_config(self):
        """Show and optionally modify model configuration."""
        models = self.config_manager.get_available_models()

        # Select model to configure
        self.console.print("\n[bold]Select model to configure:[/bold]")
        for i, (name, display_name) in enumerate(models.items(), 1):
            self.console.print(f"  {i}. {display_name} ({name})")

        choice = Prompt.ask(
            "\nEnter model number or name",
            choices=[str(i) for i in range(1, len(models) + 1)] + list(models.keys()),
            default="1"
        )

        # Convert number to model name if needed
        if choice.isdigit():
            model_name = list(models.keys())[int(choice) - 1]
        else:
            model_name = choice

        # Get model configuration
        config = self.config_manager.get_model_config(model_name)

        # Display current configuration
        self.console.print(f"\n[bold green]Configuration for {config['display_name']}:[/bold green]")

        table = Table(title="Current Settings")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="yellow")

        # Show generation config
        gen_config = config.get('generation_config', {})
        for key, value in gen_config.items():
            table.add_row(key, str(value))

        self.console.print(table)

        # Ask if user wants to modify
        modify = Prompt.ask("\nModify configuration?", choices=["y", "n"], default="n")
        if modify.lower() == 'y':
            self.console.print("\n[yellow]Configuration modification not implemented yet.[/yellow]")
            self.console.print("[dim]This feature will be added in a future update.[/dim]")

    def list_conversations(self):
        """List saved conversations."""
        conversations = self.conversation_manager.list_saved_conversations()

        if not conversations:
            self.console.print("[yellow]No saved conversations found.[/yellow]")
            return

        table = Table(title="Saved Conversations")
        table.add_column("Filename", style="cyan")
        table.add_column("Created", style="green")
        table.add_column("Messages", style="yellow")
        table.add_column("Size", style="dim")

        for conv in conversations:
            size_kb = conv['file_size'] / 1024
            table.add_row(
                conv['filename'],
                conv['created_at'][:19].replace('T', ' '),  # Format datetime
                str(conv['total_messages']),
                f"{size_kb:.1f} KB"
            )

        self.console.print(table)
    
    def run_chat_loop(self):
        """Run the main chat loop."""
        self.console.print("\n[green]Starting chat session...[/green]")
        
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]", default="").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input[1:].lower()
                    
                    if command == 'help':
                        self.show_welcome()
                    elif command == 'models':
                        self.show_models()
                    elif command == 'switch':
                        self.switch_model()
                    elif command == 'info':
                        self.show_model_info()
                    elif command == 'config':
                        self.show_config()
                    elif command == 'clear':
                        self.conversation_manager.clear_history()
                        self.console.print("[green]✓ Conversation history cleared[/green]")
                    elif command == 'save':
                        filename = Prompt.ask("Enter filename", default="conversation.json")
                        if self.conversation_manager.save_conversation(filename):
                            self.console.print(f"[green]✓ Conversation saved to {filename}[/green]")
                        else:
                            self.console.print("[red]✗ Failed to save conversation[/red]")
                    elif command == 'load':
                        filename = Prompt.ask("Enter filename", default="conversation.json")
                        if self.conversation_manager.load_conversation(filename):
                            self.console.print(f"[green]✓ Conversation loaded from {filename}[/green]")
                        else:
                            self.console.print("[red]✗ Failed to load conversation[/red]")
                    elif command == 'list':
                        self.list_conversations()
                    elif command in ['quit', 'exit', 'q']:
                        self.console.print("[yellow]Goodbye! 👋[/yellow]")
                        break
                    else:
                        self.console.print(f"[red]Unknown command: {command}[/red]")
                    
                    continue
                
                # Check if model is loaded
                if self.model_manager.current_model is None:
                    self.console.print("[yellow]Loading default model...[/yellow]")
                    if not self.model_manager.load_model():
                        self.console.print("[red]Failed to load model. Please try switching to a different model.[/red]")
                        continue
                
                # Generate response
                self.console.print("\n[dim]🤖 Thinking...[/dim]")
                
                # Add user message to conversation
                self.conversation_manager.add_message("user", user_input)
                
                # Get conversation context
                context = self.conversation_manager.get_conversation_context()
                
                # Generate response
                response = self.model_manager.generate_response(context)
                
                # Add assistant response to conversation
                self.conversation_manager.add_message("assistant", response)
                
                # Display response
                self.console.print(f"\n[bold green]🤖 Assistant[/bold green]")
                self.console.print(Panel(Markdown(response), border_style="green"))
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Chat interrupted. Use /quit to exit.[/yellow]")
            except Exception as e:
                self.console.print(f"\n[red]Error: {e}[/red]")
                self.logger.error(f"Chat loop error: {e}")


@click.command()
@click.option('--config', '-c', default='config/models.yaml', help='Path to configuration file')
@click.option('--model', '-m', help='Model to load initially')
def main(config, model):
    """Simple CLI Chatbot powered by Hugging Face Transformers."""
    cli = ChatbotCLI()
    
    # Initialize
    if not cli.initialize(config):
        sys.exit(1)
    
    # Show welcome message
    cli.show_welcome()
    
    # Load initial model if specified
    if model:
        cli.console.print(f"\n[yellow]Loading specified model: {model}...[/yellow]")
        if not cli.model_manager.load_model(model):
            cli.console.print(f"[red]Failed to load model: {model}[/red]")
    
    # Start chat loop
    cli.run_chat_loop()


if __name__ == '__main__':
    main()
