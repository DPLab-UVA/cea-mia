"""CLI commands for nanobot-lite."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from nanobot import __version__, __logo__

app = typer.Typer(
    name="nanobot",
    help=f"{__logo__} nanobot-lite - Local AI Agent",
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool):
    if value:
        console.print(f"{__logo__} nanobot-lite v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    pass


# ── onboard ──────────────────────────────────────────────────────────────────


@app.command()
def onboard():
    """初始化 nanobot 配置和工作空间。"""
    from nanobot.config import CONFIG_PATH, Config, save_config

    if CONFIG_PATH.exists():
        console.print(f"[yellow]配置文件已存在: {CONFIG_PATH}[/yellow]")
        if not typer.confirm("覆盖?"):
            raise typer.Exit()

    config = Config()
    save_config(config)
    console.print(f"[green]✓[/green] 创建配置文件: {CONFIG_PATH}")

    # Create workspace
    ws = config.workspace_path
    ws.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]✓[/green] 创建工作空间: {ws}")

    # Templates
    _create_templates(ws)

    console.print(f"\n{__logo__} nanobot-lite 已就绪!")
    console.print("\n下一步:")
    console.print(f"  1. 编辑 [cyan]{CONFIG_PATH}[/cyan] 配置你的本地模型")
    console.print("  2. 聊天: [cyan]python -m nanobot chat[/cyan]")


def _create_templates(ws: Path):
    agents = ws / "AGENTS.md"
    if not agents.exists():
        agents.write_text(
            "# Agent Instructions\n\n"
            "You are a helpful AI assistant. Be concise, accurate, and friendly.\n"
        )
        console.print("  [dim]Created AGENTS.md[/dim]")

    mem_dir = ws / "memory"
    mem_dir.mkdir(exist_ok=True)
    mem_file = mem_dir / "MEMORY.md"
    if not mem_file.exists():
        mem_file.write_text("# Long-term Memory\n\n")
        console.print("  [dim]Created memory/MEMORY.md[/dim]")


# ── chat ─────────────────────────────────────────────────────────────────────


@app.command()
def chat(
    message: str = typer.Option(None, "--message", "-m", help="发送单条消息"),
    session: str = typer.Option("default", "--session", "-s", help="会话 ID"),
    model: str = typer.Option(None, "--model", help="覆盖配置中的模型名"),
):
    """与 Agent 对话（交互模式或单条消息）。"""
    from nanobot.config import load_config
    from nanobot.agent import LocalLLMProvider
    from nanobot.agent.loop import AgentLoop

    config = load_config()
    model_name = model or config.model
    provider = LocalLLMProvider(
        api_base=config.api_base,
        api_key=config.api_key,
        model=model_name,
        timeout=120.0
    )

    ws = config.workspace_path
    ws.mkdir(parents=True, exist_ok=True)

    agent = AgentLoop(
        provider=provider,
        workspace=ws,
        max_iterations=config.max_iterations,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    if message:
        # Single message mode
        async def run_once():
            resp = await agent.process(message, session)
            console.print()
            console.print(Panel(Markdown(resp), title=f"{__logo__} nanobot", border_style="green"))

        asyncio.run(run_once())
    else:
        # Interactive mode
        _run_interactive(agent, session, model_name)


def _run_interactive(agent, session_key: str, model_name: str):
    """Run interactive TUI chat loop."""
    console.print(
        Panel(
            f"[bold]nanobot-lite[/bold] v{__version__}\n"
            f"模型: [cyan]{model_name}[/cyan]\n"
            f"会话: [cyan]{session_key}[/cyan]\n"
            f"工具: [cyan]{', '.join(agent.tools.tool_names)}[/cyan]\n\n"
            f"[dim]输入消息开始聊天, Ctrl+C 退出, /clear 清除会话, /help 查看帮助[/dim]",
            title=f"{__logo__} nanobot",
            border_style="blue",
        )
    )
    console.print()

    async def loop():
        while True:
            try:
                user_input = console.input("[bold cyan]You >[/bold cyan] ")
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]再见! 👋[/dim]")
                break

            text = user_input.strip()
            if not text:
                continue

            # Slash commands
            if text.startswith("/"):
                if _handle_slash(text, agent, session_key):
                    continue
                else:
                    break

            # Call agent
            console.print("[dim]思考中...[/dim]", end="")
            try:
                resp = await agent.process(text, session_key)
                # Clear "thinking" line
                console.print("\r", end="")
                console.print(
                    Panel(Markdown(resp), title=f"{__logo__}", border_style="green", padding=(0, 1))
                )
                console.print()
            except Exception as e:
                console.print(f"\r[red]Error: {e}[/red]")
                console.print()

    asyncio.run(loop())


def _handle_slash(cmd: str, agent, session_key: str) -> bool:
    """Handle slash commands. Returns True to continue, False to exit."""
    cmd = cmd.lower().strip()

    if cmd in ("/quit", "/exit", "/q"):
        console.print("[dim]再见! 👋[/dim]")
        return False

    if cmd == "/clear":
        s = agent.sessions.get_or_create(session_key)
        s.clear()
        agent.sessions.save(s)
        console.print("[green]✓[/green] 会话已清除\n")
        return True

    if cmd == "/help":
        console.print(
            Panel(
                "/clear   - 清除当前会话\n"
                "/quit    - 退出\n"
                "/tools   - 查看可用工具\n"
                "/status  - 查看状态\n"
                "/memory  - 查看 PMC 记忆系统状态\n"
                "/recall <query> - 测试记忆召回",
                title="命令帮助",
                border_style="yellow",
            )
        )
        console.print()
        return True

    if cmd == "/tools":
        for name in agent.tools.tool_names:
            console.print(f"  🔧 {name}")
        console.print()
        return True

    if cmd == "/status":
        from nanobot.config import load_config
        config = load_config()
        console.print(f"  模型: {config.model}")
        console.print(f"  API:  {config.api_base}")
        console.print(f"  工作空间: {config.workspace_path}")
        console.print()
        return True

    if cmd == "/memory":
        _show_memory_stats(agent, console)
        return True

    if cmd.startswith("/recall"):
        query = cmd[len("/recall"):].strip()
        if query:
            _show_recall(agent, query, console)
        else:
            console.print("[yellow]用法: /recall <查询词>[/yellow]\n")
        return True

    console.print(f"[yellow]未知命令: {cmd}[/yellow] (输入 /help 查看帮助)\n")
    return True


def _show_memory_stats(agent, console):
    """Display PMC memory system statistics."""
    try:
        stats = agent.pmc.stats()
        console.print(
            Panel(
                f"📝 情景记忆 (Episodic):  {stats.get('episodic', 0)} 条 "
                f"({stats.get('unconsolidated_episodes', 0)} 待巩固)\n"
                f"🧠 语义记忆 (Semantic):  {stats.get('semantic', 0)} 条 "
                f"({stats.get('active_semantic', 0)} 活跃)\n"
                f"⚡ 程序记忆 (Procedural): {stats.get('procedural', 0)} 条 "
                f"({stats.get('active_procedural', 0)} 活跃)",
                title="🧠 PMC 记忆系统",
                border_style="magenta",
            )
        )

        # Show top semantic memories
        semantic = agent.pmc.store.get_active_semantic()[:5]
        if semantic:
            console.print("\n[bold]Top 语义记忆:[/bold]")
            for s in semantic:
                console.print(
                    f"  [{s.confidence:.0%}] {s.content[:80]} "
                    f"[dim](reinforced {s.reinforcement_count}x)[/dim]"
                )

        # Show procedural strategies
        procedural = agent.pmc.store.get_active_procedural()[:5]
        if procedural:
            console.print("\n[bold]策略记忆:[/bold]")
            for p in procedural:
                console.print(
                    f"  [{p.confidence:.0%}] {p.trigger[:50]} → {p.action[:50]}"
                )

        console.print()
    except Exception as e:
        console.print(f"[red]Error reading memory: {e}[/red]\n")


def _show_recall(agent, query, console):
    """Test memory recall for a given query."""
    try:
        result = agent.pmc.recall(query)
        if result.is_empty():
            console.print("[dim]没有找到相关记忆[/dim]\n")
            return

        text = result.format_for_prompt()
        console.print(Panel(Markdown(text), title=f"🔍 Recall: {query}", border_style="cyan"))
        console.print()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]\n")


# ── status ───────────────────────────────────────────────────────────────────


@app.command()
def status():
    """查看 nanobot 状态。"""
    from nanobot.config import CONFIG_PATH, load_config

    config = load_config()
    ws = config.workspace_path

    console.print(f"\n{__logo__} nanobot-lite Status\n")
    console.print(f"  配置文件: {CONFIG_PATH} {'[green]✓[/green]' if CONFIG_PATH.exists() else '[red]✗[/red]'}")
    console.print(f"  工作空间: {ws} {'[green]✓[/green]' if ws.exists() else '[red]✗[/red]'}")
    console.print(f"  模型: [cyan]{config.model}[/cyan]")
    console.print(f"  API:  [cyan]{config.api_base}[/cyan]")
    console.print(f"  Max iterations: {config.max_iterations}")
    console.print(f"  Temperature: {config.temperature}")
    console.print()

    # Quick connectivity check
    import httpx
    try:
        r = httpx.get(f"{config.api_base}/models", timeout=5.0, headers={"Authorization": f"Bearer {config.api_key}"})
        if r.status_code == 200:
            models = r.json().get("data", [])
            console.print(f"  [green]✓ 模型服务可用[/green] ({len(models)} models)")
            for m in models[:5]:
                console.print(f"    - {m.get('id', '?')}")
            if len(models) > 5:
                console.print(f"    ... 等 {len(models)} 个模型")
        else:
            console.print(f"  [yellow]⚠ 模型服务返回 {r.status_code}[/yellow]")
    except Exception as e:
        console.print(f"  [red]✗ 无法连接模型服务: {e}[/red]")
    console.print()


if __name__ == "__main__":
    app()
