import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { basename, dirname, extname, join } from "node:path";
import { homedir } from "node:os";

import type { ImageContent, TextContent } from "@mariozechner/pi-ai";
import type { AgentMessage } from "@mariozechner/pi-agent-core";
import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";
import { Type } from "@sinclair/typebox";

interface TelegramConfig {
	botToken?: string;
	botUsername?: string;
	botId?: number;
	allowedUserId?: number;
	lastUpdateId?: number;
}

interface TelegramApiResponse<T> {
	ok: boolean;
	result?: T;
	description?: string;
	error_code?: number;
}

interface TelegramUser {
	id: number;
	is_bot: boolean;
	first_name: string;
	username?: string;
}

interface TelegramChat {
	id: number;
	type: string;
}

interface TelegramPhotoSize {
	file_id: string;
	file_size?: number;
}

interface TelegramDocument {
	file_id: string;
	file_name?: string;
	mime_type?: string;
	file_size?: number;
}

interface TelegramVideo {
	file_id: string;
	file_name?: string;
	mime_type?: string;
	file_size?: number;
}

interface TelegramAudio {
	file_id: string;
	file_name?: string;
	mime_type?: string;
	file_size?: number;
}

interface TelegramVoice {
	file_id: string;
	mime_type?: string;
	file_size?: number;
}

interface TelegramAnimation {
	file_id: string;
	file_name?: string;
	mime_type?: string;
	file_size?: number;
}

interface TelegramSticker {
	file_id: string;
	emoji?: string;
}

interface TelegramFileInfo {
	file_id: string;
	fileName: string;
	mimeType?: string;
	isImage: boolean;
}

interface TelegramMessage {
	message_id: number;
	chat: TelegramChat;
	from?: TelegramUser;
	text?: string;
	caption?: string;
	media_group_id?: string;
	photo?: TelegramPhotoSize[];
	document?: TelegramDocument;
	video?: TelegramVideo;
	audio?: TelegramAudio;
	voice?: TelegramVoice;
	animation?: TelegramAnimation;
	sticker?: TelegramSticker;
}

interface TelegramUpdate {
	update_id: number;
	message?: TelegramMessage;
	edited_message?: TelegramMessage;
	callback_query?: TelegramCallbackQuery;
}

interface TelegramCallbackQuery {
	id: string;
	from: TelegramUser;
	message?: {
		message_id: number;
		chat: TelegramChat;
		text?: string;
	};
	data?: string;
}

interface TelegramCallbackAnswer {
	ok: boolean;
}

interface TelegramGetFileResult {
	file_path: string;
}

interface TelegramSentMessage {
	message_id: number;
}

interface DownloadedTelegramFile {
	path: string;
	fileName: string;
	isImage: boolean;
	mimeType?: string;
}

interface PendingTelegramTurn {
	chatId: number;
	replyToMessageId: number;
	queuedAttachments: QueuedAttachment[];
	content: Array<TextContent | ImageContent>;
	historyText: string;
}

type ActiveTelegramTurn = PendingTelegramTurn;

interface QueuedAttachment {
	path: string;
	fileName: string;
}

interface TelegramPreviewState {
	mode: "draft" | "message";
	draftId?: number;
	messageId?: number;
	pendingText: string;
	lastSentText: string;
	flushTimer?: ReturnType<typeof setTimeout>;
}

interface TelegramMediaGroupState {
	messages: TelegramMessage[];
	flushTimer?: ReturnType<typeof setTimeout>;
}

// PI_TELEGRAM_CONFIG overrides the config file path so dev/test bots can run
// alongside a prod bot on the same box without clobbering the default config.
const CONFIG_PATH = process.env.PI_TELEGRAM_CONFIG ?? join(homedir(), ".pi", "agent", "telegram.json");
const TEMP_DIR = join(homedir(), ".pi", "agent", "tmp", "telegram");
const TELEGRAM_PREFIX = "[telegram]";
const MAX_MESSAGE_LENGTH = 4096;
const MAX_ATTACHMENTS_PER_TURN = 10;
const PREVIEW_THROTTLE_MS = 750;
const TELEGRAM_DRAFT_ID_MAX = 2_147_483_647;
const TELEGRAM_MEDIA_GROUP_DEBOUNCE_MS = 1200;

const SYSTEM_PROMPT_SUFFIX = `

Telegram bridge extension is active.
- Messages forwarded from Telegram are prefixed with "[telegram]".
- [telegram] messages may include local temp file paths for Telegram attachments. Read those files as needed.
- If a [telegram] user asked for a file or generated artifact, use the telegram_attach tool with the local file path so the extension can send it with your next final reply.
- Do not assume mentioning a local file path in plain text will send it to Telegram. Use telegram_attach.`;

function isTelegramPrompt(prompt: string): boolean {
	return prompt.trimStart().startsWith(TELEGRAM_PREFIX);
}

function sanitizeFileName(name: string): string {
	return name.replace(/[^a-zA-Z0-9._-]+/g, "_");
}

function guessExtensionFromMime(mimeType: string | undefined, fallback: string): string {
	if (!mimeType) return fallback;
	const normalized = mimeType.toLowerCase();
	if (normalized === "image/jpeg") return ".jpg";
	if (normalized === "image/png") return ".png";
	if (normalized === "image/webp") return ".webp";
	if (normalized === "image/gif") return ".gif";
	if (normalized === "audio/ogg") return ".ogg";
	if (normalized === "audio/mpeg") return ".mp3";
	if (normalized === "audio/wav") return ".wav";
	if (normalized === "video/mp4") return ".mp4";
	if (normalized === "application/pdf") return ".pdf";
	return fallback;
}

function guessMediaType(path: string): string | undefined {
	const ext = extname(path).toLowerCase();
	if (ext === ".jpg" || ext === ".jpeg") return "image/jpeg";
	if (ext === ".png") return "image/png";
	if (ext === ".webp") return "image/webp";
	if (ext === ".gif") return "image/gif";
	return undefined;
}

function isImageMimeType(mimeType: string | undefined): boolean {
	return mimeType?.toLowerCase().startsWith("image/") ?? false;
}

function formatTokens(count: number): string {
	if (count < 1000) return count.toString();
	if (count < 10000) return `${(count / 1000).toFixed(1)}k`;
	if (count < 1000000) return `${Math.round(count / 1000)}k`;
	if (count < 10000000) return `${(count / 1000000).toFixed(1)}M`;
	return `${Math.round(count / 1000000)}M`;
}

function chunkParagraphs(text: string): string[] {
	if (text.length <= MAX_MESSAGE_LENGTH) return [text];

	const normalized = text.replace(/\r\n/g, "\n");
	const paragraphs = normalized.split(/\n\n+/);
	const chunks: string[] = [];
	let current = "";

	const flushCurrent = (): void => {
		if (current.trim().length > 0) chunks.push(current);
		current = "";
	};

	const splitLongBlock = (block: string): string[] => {
		if (block.length <= MAX_MESSAGE_LENGTH) return [block];
		const lines = block.split("\n");
		const lineChunks: string[] = [];
		let lineCurrent = "";
		for (const line of lines) {
			const candidate = lineCurrent.length === 0 ? line : `${lineCurrent}\n${line}`;
			if (candidate.length <= MAX_MESSAGE_LENGTH) {
				lineCurrent = candidate;
				continue;
			}
			if (lineCurrent.length > 0) {
				lineChunks.push(lineCurrent);
				lineCurrent = "";
			}
			if (line.length <= MAX_MESSAGE_LENGTH) {
				lineCurrent = line;
				continue;
			}
			for (let i = 0; i < line.length; i += MAX_MESSAGE_LENGTH) {
				lineChunks.push(line.slice(i, i + MAX_MESSAGE_LENGTH));
			}
		}
		if (lineCurrent.length > 0) lineChunks.push(lineCurrent);
		return lineChunks;
	};

	for (const paragraph of paragraphs) {
		if (paragraph.length === 0) continue;
		const parts = splitLongBlock(paragraph);
		for (const part of parts) {
			const candidate = current.length === 0 ? part : `${current}\n\n${part}`;
			if (candidate.length <= MAX_MESSAGE_LENGTH) {
				current = candidate;
			} else {
				flushCurrent();
				current = part;
			}
		}
	}
	flushCurrent();
	return chunks;
}

async function readConfig(): Promise<TelegramConfig> {
	try {
		const content = await readFile(CONFIG_PATH, "utf8");
		const parsed = JSON.parse(content) as TelegramConfig;
		return parsed;
	} catch {
		return {};
	}
}

async function writeConfig(config: TelegramConfig): Promise<void> {
	await mkdir(dirname(CONFIG_PATH), { recursive: true });
	await writeFile(CONFIG_PATH, JSON.stringify(config, null, "\t") + "\n", "utf8");
}

export default function (pi: ExtensionAPI) {
	let config: TelegramConfig = {};
	let pollingController: AbortController | undefined;
	let pollingPromise: Promise<void> | undefined;
	let queuedTelegramTurns: PendingTelegramTurn[] = [];
	let activeTelegramTurn: ActiveTelegramTurn | undefined;
	let typingInterval: ReturnType<typeof setInterval> | undefined;
	let currentAbort: (() => void) | undefined;
	let preserveQueuedTurnsAsHistory = false;
	let setupInProgress = false;
	let previewState: TelegramPreviewState | undefined;
	let draftSupport: "unknown" | "supported" | "unsupported" = "unknown";
	let nextDraftId = 0;
	const mediaGroups = new Map<string, TelegramMediaGroupState>();
	// chatId -> { models, messageId, page } for pending model picker keyboards
	const pendingModelSelections = new Map<
		number,
		{
			models: Array<{
				ref: string;
				provider: string;
				id: string;
				name?: string;
			}>;
			messageId: number;
			page: number;
		}
	>();

	function allocateDraftId(): number {
		nextDraftId = nextDraftId >= TELEGRAM_DRAFT_ID_MAX ? 1 : nextDraftId + 1;
		return nextDraftId;
	}

	function updateStatus(ctx: ExtensionContext, error?: string): void {
		// ctx may be stale if the session was replaced (e.g. /new triggered via
		// pi.executeCommand) while the polling lifecycle still holds a reference.
		// Touching a stale ctx throws; the status indicator is decorative, so
		// swallow the error rather than tearing pi down.
		try {
			const theme = ctx.ui.theme;
			const label = theme.fg("accent", "telegram");
			if (error) {
				ctx.ui.setStatus("telegram", `${label} ${theme.fg("error", "error")} ${theme.fg("muted", error)}`);
				return;
			}
			if (!config.botToken) {
				ctx.ui.setStatus("telegram", `${label} ${theme.fg("muted", "not configured")}`);
				return;
			}
			if (!pollingPromise) {
				ctx.ui.setStatus("telegram", `${label} ${theme.fg("muted", "disconnected")}`);
				return;
			}
			if (!config.allowedUserId) {
				ctx.ui.setStatus("telegram", `${label} ${theme.fg("warning", "awaiting pairing")}`);
				return;
			}
			if (activeTelegramTurn || queuedTelegramTurns.length > 0) {
				const queued = queuedTelegramTurns.length > 0 ? theme.fg("muted", ` +${queuedTelegramTurns.length} queued`) : "";
				ctx.ui.setStatus("telegram", `${label} ${theme.fg("accent", "processing")}${queued}`);
				return;
			}
			ctx.ui.setStatus("telegram", `${label} ${theme.fg("success", "connected")}`);
		} catch {
			// ctx is stale after a session replacement — ignore.
		}
	}

	async function callTelegram<TResponse>(
		method: string,
		body: Record<string, unknown>,
		options?: { signal?: AbortSignal },
	): Promise<TResponse> {
		if (!config.botToken) throw new Error("Telegram bot token is not configured");
		const response = await fetch(`https://api.telegram.org/bot${config.botToken}/${method}`, {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify(body),
			signal: options?.signal,
		});
			const data = (await response.json()) as TelegramApiResponse<TResponse>;
		if (!data.ok || data.result === undefined) {
			throw new Error(data.description || `Telegram API ${method} failed`);
		}
		return data.result;
	}

	async function callTelegramMultipart<TResponse>(
		method: string,
		fields: Record<string, string>,
		fileField: string,
		filePath: string,
		fileName: string,
		options?: { signal?: AbortSignal },
	): Promise<TResponse> {
		if (!config.botToken) throw new Error("Telegram bot token is not configured");
		const form = new FormData();
		for (const [key, value] of Object.entries(fields)) {
			form.set(key, value);
		}
		const buffer = await readFile(filePath);
		form.set(fileField, new Blob([buffer]), fileName);
		const response = await fetch(`https://api.telegram.org/bot${config.botToken}/${method}`, {
			method: "POST",
			body: form,
			signal: options?.signal,
		});
		const data = (await response.json()) as TelegramApiResponse<TResponse>;
		if (!data.ok || data.result === undefined) {
			throw new Error(data.description || `Telegram API ${method} failed`);
		}
		return data.result;
	}

	let lastSyncedCommandsHash: string | undefined;
	let activeReverseMap: Map<string, string> = new Map();

	function sanitizeTelegramCommandName(raw: string): string | undefined {
		const lowered = raw.toLowerCase().replace(/[^a-z0-9_]+/g, "_").replace(/^_+|_+$/g, "");
		if (!lowered) return undefined;
		return lowered.slice(0, 32);
	}

	// pi's built-in slash commands are NOT returned by pi.getCommands() — that
	// only lists extension-registered commands, prompt templates, and skills.
	// Mirrored from @mariozechner/pi-coding-agent/dist/core/slash-commands.js so
	// the Telegram menu surfaces the full command set users see in the pi TUI.
	const PI_BUILTIN_COMMANDS: ReadonlyArray<{ name: string; description: string }> = [
		{ name: "settings", description: "Open settings menu" },
		{ name: "model", description: "Select model" },
		{ name: "scoped-models", description: "Enable/disable models for Ctrl+P cycling" },
		{ name: "export", description: "Export session (HTML or JSONL)" },
		{ name: "import", description: "Import and resume a session from a JSONL file" },
		{ name: "share", description: "Share session as a secret GitHub gist" },
		{ name: "copy", description: "Copy last agent message to clipboard" },
		{ name: "name", description: "Set session display name" },
		{ name: "session", description: "Show session info and stats" },
		{ name: "changelog", description: "Show changelog entries" },
		{ name: "hotkeys", description: "Show all keyboard shortcuts" },
		{ name: "fork", description: "Create a new fork from a previous user message" },
		{ name: "clone", description: "Duplicate the current session at the current position" },
		{ name: "tree", description: "Navigate session tree (switch branches)" },
		{ name: "login", description: "Configure provider authentication" },
		{ name: "logout", description: "Remove provider authentication" },
		{ name: "new", description: "Start a new session" },
		{ name: "compact", description: "Manually compact the session context" },
		{ name: "resume", description: "Resume a different session" },
		{ name: "reload", description: "Reload keybindings, extensions, skills, prompts, and themes" },
		{ name: "quit", description: "Quit pi" },
	];

	function buildTelegramCommandList(): { payload: Array<{ command: string; description: string }>; reverseMap: Map<string, string> } {
		const seen = new Set<string>();
		const payload: Array<{ command: string; description: string }> = [];
		const reverseMap = new Map<string, string>();

		const addEntry = (originalName: string, description: string): void => {
			const tgName = sanitizeTelegramCommandName(originalName);
			if (!tgName || seen.has(tgName)) return;
			const desc = (description ?? "pi command").slice(0, 256);
			payload.push({ command: tgName, description: desc });
			reverseMap.set(tgName, originalName);
			seen.add(tgName);
		};

		// Telegram conventions plus Telegram-only handlers — these always run from Telegram.
		payload.push({ command: "start", description: "Pair this Telegram account with the bot" });
		payload.push({ command: "help", description: "Show Telegram bridge help" });
		payload.push({ command: "stop", description: "Abort the current pi turn" });
		for (const e of payload) seen.add(e.command);

		// pi's built-ins (model, compact, new, fork, …)
		for (const cmd of PI_BUILTIN_COMMANDS) {
			if (payload.length >= 100) break;
			addEntry(cmd.name, cmd.description);
		}

		// Extension-registered commands and prompt templates (skips skills — they use `:`).
		for (const cmd of pi.getCommands()) {
			if (payload.length >= 100) break;
			if (cmd.source === "skill") continue;
			if (cmd.name.includes(":")) continue;
			addEntry(cmd.name, cmd.description ?? "pi command");
		}

		return { payload, reverseMap };
	}

	async function syncTelegramCommands(ctx: ExtensionContext): Promise<void> {
		if (!config.botToken) return;
		const { payload, reverseMap } = buildTelegramCommandList();
		activeReverseMap = reverseMap;
		const hash = JSON.stringify(payload);
		if (hash === lastSyncedCommandsHash) return;
		try {
			await callTelegram("setMyCommands", {
				commands: payload,
				scope: { type: "all_private_chats" },
			});
			lastSyncedCommandsHash = hash;
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			updateStatus(ctx, `setMyCommands failed: ${message}`);
		}
	}

	async function downloadTelegramFile(fileId: string, suggestedName: string): Promise<string> {
		if (!config.botToken) throw new Error("Telegram bot token is not configured");
		const file = await callTelegram<TelegramGetFileResult>("getFile", { file_id: fileId });
		await mkdir(TEMP_DIR, { recursive: true });
		const targetPath = join(TEMP_DIR, `${Date.now()}-${sanitizeFileName(suggestedName)}`);
		const response = await fetch(`https://api.telegram.org/file/bot${config.botToken}/${file.file_path}`);
		if (!response.ok) throw new Error(`Failed to download Telegram file: ${response.status}`);
		const arrayBuffer = await response.arrayBuffer();
		await writeFile(targetPath, Buffer.from(arrayBuffer));
		return targetPath;
	}

	function startTypingLoop(ctx: ExtensionContext, chatId?: number): void {
		const targetChatId = chatId ?? activeTelegramTurn?.chatId;
		if (typingInterval || targetChatId === undefined) return;

		const sendTyping = async (): Promise<void> => {
			try {
				await callTelegram("sendChatAction", { chat_id: targetChatId, action: "typing" });
			} catch (error) {
				const message = error instanceof Error ? error.message : String(error);
				updateStatus(ctx, `typing failed: ${message}`);
			}
		};

		void sendTyping();
		typingInterval = setInterval(() => {
			void sendTyping();
		}, 4000);
	}

	function stopTypingLoop(): void {
		if (!typingInterval) return;
		clearInterval(typingInterval);
		typingInterval = undefined;
	}

	function isAssistantMessage(message: AgentMessage): boolean {
		return (message as unknown as { role?: string }).role === "assistant";
	}

	function getMessageText(message: AgentMessage): string {
		const value = message as unknown as Record<string, unknown>;
		const content = Array.isArray(value.content) ? value.content : [];
		return content
			.filter((block): block is { type: string; text?: string } => typeof block === "object" && block !== null && "type" in block)
			.filter((block) => block.type === "text" && typeof block.text === "string")
			.map((block) => block.text as string)
			.join("")
			.trim();
	}

	async function clearPreview(chatId: number): Promise<void> {
		const state = previewState;
		if (!state) return;
		if (state.flushTimer) {
			clearTimeout(state.flushTimer);
			state.flushTimer = undefined;
		}
		previewState = undefined;
		if (state.mode === "draft" && state.draftId !== undefined) {
			try {
				await callTelegram("sendMessageDraft", { chat_id: chatId, draft_id: state.draftId, text: "" });
			} catch {
				// ignore
			}
		}
	}

	async function flushPreview(chatId: number): Promise<void> {
		const state = previewState;
		if (!state) return;
		state.flushTimer = undefined;
		const text = state.pendingText.trim();
		if (!text || text === state.lastSentText) return;
		const truncated = text.length > MAX_MESSAGE_LENGTH ? text.slice(0, MAX_MESSAGE_LENGTH) : text;

		if (draftSupport !== "unsupported") {
			const draftId = state.draftId ?? allocateDraftId();
			state.draftId = draftId;
			try {
				await callTelegram("sendMessageDraft", { chat_id: chatId, draft_id: draftId, text: truncated });
				draftSupport = "supported";
				state.mode = "draft";
				state.lastSentText = truncated;
				return;
			} catch {
				draftSupport = "unsupported";
			}
		}

		if (state.messageId === undefined) {
			const sent = await callTelegram<TelegramSentMessage>("sendMessage", { chat_id: chatId, text: truncated });
			state.messageId = sent.message_id;
			state.mode = "message";
			state.lastSentText = truncated;
			return;
		}
		await callTelegram("editMessageText", { chat_id: chatId, message_id: state.messageId, text: truncated });
		state.mode = "message";
		state.lastSentText = truncated;
	}

	function schedulePreviewFlush(chatId: number): void {
		if (!previewState || previewState.flushTimer) return;
		previewState.flushTimer = setTimeout(() => {
			void flushPreview(chatId);
		}, PREVIEW_THROTTLE_MS);
	}

	async function finalizePreview(chatId: number): Promise<boolean> {
		const state = previewState;
		if (!state) return false;
		await flushPreview(chatId);
		const finalText = (state.pendingText.trim() || state.lastSentText).trim();
		if (!finalText) {
			await clearPreview(chatId);
			return false;
		}
		if (state.mode === "draft") {
			await callTelegram<TelegramSentMessage>("sendMessage", { chat_id: chatId, text: finalText });
			await clearPreview(chatId);
			return true;
		}
		previewState = undefined;
		return state.messageId !== undefined;
	}

	async function sendTextReply(chatId: number, _replyToMessageId: number, text: string): Promise<number | undefined> {
		const chunks = chunkParagraphs(text);
		let lastMessageId: number | undefined;
		for (const chunk of chunks) {
			const sent = await callTelegram<TelegramSentMessage>("sendMessage", {
				chat_id: chatId,
				text: chunk,
			});
			lastMessageId = sent.message_id;
		}
		return lastMessageId;
	}

	async function sendQueuedAttachments(turn: ActiveTelegramTurn): Promise<void> {
		for (const attachment of turn.queuedAttachments) {
			try {
				const mediaType = guessMediaType(attachment.path);
				const method = mediaType ? "sendPhoto" : "sendDocument";
				const fieldName = mediaType ? "photo" : "document";
				await callTelegramMultipart<TelegramSentMessage>(
					method,
					{
						chat_id: String(turn.chatId),
					},
					fieldName,
					attachment.path,
					attachment.fileName,
				);
			} catch (error) {
				const message = error instanceof Error ? error.message : String(error);
				await sendTextReply(turn.chatId, turn.replyToMessageId, `Failed to send attachment ${attachment.fileName}: ${message}`);
			}
		}
	}

	function extractAssistantText(messages: AgentMessage[]): { text?: string; stopReason?: string; errorMessage?: string } {
		for (let i = messages.length - 1; i >= 0; i--) {
			const message = messages[i] as unknown as Record<string, unknown>;
			if (message.role !== "assistant") continue;
			const stopReason = typeof message.stopReason === "string" ? message.stopReason : undefined;
			const errorMessage = typeof message.errorMessage === "string" ? message.errorMessage : undefined;
			const content = Array.isArray(message.content) ? message.content : [];
			const text = content
				.filter((block): block is { type: string; text?: string } => typeof block === "object" && block !== null && "type" in block)
				.filter((block) => block.type === "text" && typeof block.text === "string")
				.map((block) => block.text as string)
				.join("")
				.trim();
			return { text: text || undefined, stopReason, errorMessage };
		}
		return {};
	}

	function collectTelegramFileInfos(messages: TelegramMessage[]): TelegramFileInfo[] {
		const files: TelegramFileInfo[] = [];
		for (const message of messages) {
			if (Array.isArray(message.photo) && message.photo.length > 0) {
				const photo = [...message.photo].sort((a, b) => (a.file_size ?? 0) - (b.file_size ?? 0)).pop();
				if (photo) {
					files.push({
						file_id: photo.file_id,
						fileName: `photo-${message.message_id}.jpg`,
						mimeType: "image/jpeg",
						isImage: true,
					});
				}
			}
			if (message.document) {
				const fileName = message.document.file_name || `document-${message.message_id}${guessExtensionFromMime(message.document.mime_type, "")}`;
				files.push({
					file_id: message.document.file_id,
					fileName,
					mimeType: message.document.mime_type,
					isImage: isImageMimeType(message.document.mime_type),
				});
			}
			if (message.video) {
				const fileName = message.video.file_name || `video-${message.message_id}${guessExtensionFromMime(message.video.mime_type, ".mp4")}`;
				files.push({
					file_id: message.video.file_id,
					fileName,
					mimeType: message.video.mime_type,
					isImage: false,
				});
			}
			if (message.audio) {
				const fileName = message.audio.file_name || `audio-${message.message_id}${guessExtensionFromMime(message.audio.mime_type, ".mp3")}`;
				files.push({
					file_id: message.audio.file_id,
					fileName,
					mimeType: message.audio.mime_type,
					isImage: false,
				});
			}
			if (message.voice) {
				files.push({
					file_id: message.voice.file_id,
					fileName: `voice-${message.message_id}${guessExtensionFromMime(message.voice.mime_type, ".ogg")}`,
					mimeType: message.voice.mime_type,
					isImage: false,
				});
			}
			if (message.animation) {
				const fileName = message.animation.file_name || `animation-${message.message_id}${guessExtensionFromMime(message.animation.mime_type, ".mp4")}`;
				files.push({
					file_id: message.animation.file_id,
					fileName,
					mimeType: message.animation.mime_type,
					isImage: false,
				});
			}
			if (message.sticker) {
				files.push({
					file_id: message.sticker.file_id,
					fileName: `sticker-${message.message_id}.webp`,
					mimeType: "image/webp",
					isImage: true,
				});
			}
		}
		return files;
	}

	async function buildTelegramFiles(messages: TelegramMessage[]): Promise<DownloadedTelegramFile[]> {
		const downloaded: DownloadedTelegramFile[] = [];
		for (const file of collectTelegramFileInfos(messages)) {
			const path = await downloadTelegramFile(file.file_id, file.fileName);
			downloaded.push({ path, fileName: file.fileName, isImage: file.isImage, mimeType: file.mimeType });
		}
		return downloaded;
	}

	async function promptForConfig(ctx: ExtensionContext): Promise<void> {
		if (!ctx.hasUI || setupInProgress) return;
		setupInProgress = true;
		try {
			const token = await ctx.ui.input("Telegram bot token", "123456:ABCDEF...");
			if (!token) return;

			const nextConfig: TelegramConfig = { ...config, botToken: token.trim() };
			const response = await fetch(`https://api.telegram.org/bot${nextConfig.botToken}/getMe`);
			const data = (await response.json()) as TelegramApiResponse<TelegramUser>;
			if (!data.ok || !data.result) {
				ctx.ui.notify(data.description || "Invalid Telegram bot token", "error");
				return;
			}

			nextConfig.botId = data.result.id;
			nextConfig.botUsername = data.result.username;
			config = nextConfig;
			await writeConfig(config);
			ctx.ui.notify(`Telegram bot connected: @${config.botUsername ?? "unknown"}`, "info");
			ctx.ui.notify("Send /start to your bot in Telegram to pair this extension with your account.", "info");
			await startPolling(ctx);
			updateStatus(ctx);
		} finally {
			setupInProgress = false;
		}
	}

	async function stopPolling(): Promise<void> {
		stopTypingLoop();
		pollingController?.abort();
		pollingController = undefined;
		await pollingPromise?.catch(() => undefined);
		pollingPromise = undefined;
	}

	// Synchronous abort for use inside session_shutdown / nested runtime calls.
	// Awaiting pollingPromise from a handler that is itself running on the
	// polling task deadlocks (e.g. pi.executeCommand("new") triggers
	// session_shutdown which would then wait for the very poll iteration that
	// dispatched it). Detach the controller and let the loop exit on its next
	// signal-checked callTelegram.
	function abortPollingForShutdown(): void {
		stopTypingLoop();
		pollingController?.abort();
		pollingController = undefined;
		pollingPromise = undefined;
	}

	function formatTelegramHistoryText(rawText: string, files: DownloadedTelegramFile[]): string {
		let summary = rawText.length > 0 ? rawText : "(no text)";
		if (files.length > 0) {
			summary += `\nAttachments:`;
			for (const file of files) {
				summary += `\n- ${file.path}`;
			}
		}
		return summary;
	}

	async function createTelegramTurn(
		messages: TelegramMessage[],
		historyTurns: PendingTelegramTurn[] = [],
	): Promise<PendingTelegramTurn> {
		const firstMessage = messages[0];
		if (!firstMessage) throw new Error("Missing Telegram message for turn creation");
		const rawText = messages.map((message) => (message.text || message.caption || "").trim()).filter(Boolean).join("\n\n");
		const files = await buildTelegramFiles(messages);
		const content: Array<TextContent | ImageContent> = [];
		let prompt = `${TELEGRAM_PREFIX}`;

		if (historyTurns.length > 0) {
			prompt += `\n\nEarlier Telegram messages arrived after an aborted turn. Treat them as prior user messages, in order:`;
			for (const [index, turn] of historyTurns.entries()) {
				prompt += `\n\n${index + 1}. ${turn.historyText}`;
			}
			prompt += `\n\nCurrent Telegram message:`;
		}

		if (rawText.length > 0) {
			prompt += historyTurns.length > 0 ? `\n${rawText}` : ` ${rawText}`;
		}
		if (files.length > 0) {
			prompt += `\n\nTelegram attachments were saved locally:`;
			for (const file of files) {
				prompt += `\n- ${file.path}`;
			}
		}
		content.push({ type: "text", text: prompt });

		for (const file of files) {
			if (!file.isImage) continue;
			const mediaType = file.mimeType || guessMediaType(file.path);
			if (!mediaType) continue;
			const buffer = await readFile(file.path);
			content.push({
				type: "image",
				data: buffer.toString("base64"),
				mimeType: mediaType,
			});
		}

		return {
			chatId: firstMessage.chat.id,
			replyToMessageId: firstMessage.message_id,
			queuedAttachments: [],
			content,
			historyText: formatTelegramHistoryText(rawText, files),
		};
	}

	async function dispatchAuthorizedTelegramMessages(messages: TelegramMessage[], ctx: ExtensionContext): Promise<void> {
		const firstMessage = messages[0];
		if (!firstMessage) return;
		const rawText = messages.map((message) => (message.text || message.caption || "").trim()).find((text) => text.length > 0) || "";
		const lower = rawText.toLowerCase();

		if (lower === "stop" || lower === "/stop") {
			if (currentAbort) {
				if (queuedTelegramTurns.length > 0) {
					preserveQueuedTurnsAsHistory = true;
				}
				currentAbort();
				updateStatus(ctx);
				await sendTextReply(firstMessage.chat.id, firstMessage.message_id, "Aborted current turn.");
			} else {
				await sendTextReply(firstMessage.chat.id, firstMessage.message_id, "No active turn.");
			}
			return;
		}

		if (lower === "/compact") {
			if (!ctx.isIdle()) {
				await sendTextReply(firstMessage.chat.id, firstMessage.message_id, "Cannot compact while pi is busy. Send \"stop\" first.");
				return;
			}
			ctx.compact({
				onComplete: () => {
					void sendTextReply(firstMessage.chat.id, firstMessage.message_id, "Compaction completed.");
				},
				onError: (error) => {
					const message = error instanceof Error ? error.message : String(error);
					void sendTextReply(firstMessage.chat.id, firstMessage.message_id, `Compaction failed: ${message}`);
				},
			});
			await sendTextReply(firstMessage.chat.id, firstMessage.message_id, "Compaction started.");
			return;
		}

		if (lower === "/status") {
			let totalInput = 0;
			let totalOutput = 0;
			let totalCacheRead = 0;
			let totalCacheWrite = 0;
			let totalCost = 0;

			for (const entry of ctx.sessionManager.getEntries()) {
				if (entry.type !== "message" || entry.message.role !== "assistant") continue;
				totalInput += entry.message.usage.input;
				totalOutput += entry.message.usage.output;
				totalCacheRead += entry.message.usage.cacheRead;
				totalCacheWrite += entry.message.usage.cacheWrite;
				totalCost += entry.message.usage.cost.total;
			}

			const usage = ctx.getContextUsage();
			const lines: string[] = [];
			if (ctx.model) {
				lines.push(`Model: ${ctx.model.provider}/${ctx.model.id}`);
			}
			const tokenParts: string[] = [];
			if (totalInput) tokenParts.push(`↑${formatTokens(totalInput)}`);
			if (totalOutput) tokenParts.push(`↓${formatTokens(totalOutput)}`);
			if (totalCacheRead) tokenParts.push(`R${formatTokens(totalCacheRead)}`);
			if (totalCacheWrite) tokenParts.push(`W${formatTokens(totalCacheWrite)}`);
			if (tokenParts.length > 0) {
				lines.push(`Usage: ${tokenParts.join(" ")}`);
			}
			const usingSubscription = ctx.model ? ctx.modelRegistry.isUsingOAuth(ctx.model) : false;
			if (totalCost || usingSubscription) {
				lines.push(`Cost: $${totalCost.toFixed(3)}${usingSubscription ? " (sub)" : ""}`);
			}
			if (usage) {
				const contextWindow = usage.contextWindow ?? ctx.model?.contextWindow ?? 0;
				const percent = usage.percent !== null ? `${usage.percent.toFixed(1)}%` : "?";
				lines.push(`Context: ${percent}/${formatTokens(contextWindow)}`);
			} else {
				lines.push("Context: unknown");
			}
			if (lines.length === 0) {
				lines.push("No usage data yet.");
			}
			await sendTextReply(firstMessage.chat.id, firstMessage.message_id, lines.join("\n"));
			return;
		}

		if (lower === "/model" || lower.startsWith("/model ")) {
			const modelArg = lower === "/model" ? undefined : rawText.slice(7).trim();
			const chatId = firstMessage.chat.id;
			const messageId = firstMessage.message_id;

			if (modelArg) {
				// Try to match and switch to a model
				const allModels = ctx.modelRegistry.getAvailable();
				const currentRef = ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : null;

				// Try exact provider/id match first
				let matched = allModels.find(
					(m) => `${m.provider}/${m.id}`.toLowerCase() === modelArg.toLowerCase(),
				);

				// Then try exact id match
				if (!matched) {
					matched = allModels.find((m) => m.id.toLowerCase() === modelArg.toLowerCase());
				}

				// Then try partial match
				if (!matched) {
					const partial = allModels.filter(
						(m) =>
							m.provider.toLowerCase().includes(modelArg) ||
							m.id.toLowerCase().includes(modelArg) ||
							(m.name && m.name.toLowerCase().includes(modelArg)),
					);

					if (partial.length === 0) {
						await sendTextReply(
							chatId,
							messageId,
							`No models matching "${modelArg}".\n\nSend /model to list all available models.`,
						);
						return;
					}

					if (partial.length > 1) {
						const lines = [
							`Multiple models match "${modelArg}":`,
							...partial.map((m) => {
								const ref = `${m.provider}/${m.id}`;
								const suffix = m.name && m.name !== m.id ? ` (${m.name})` : "";
								return `  ${ref}${suffix}`;
							}),
							`\nBe more specific, e.g. /model ${partial[0].provider}/${partial[0].id}`,
						];
						await sendTextReply(chatId, messageId, lines.join("\n"));
						return;
					}

					matched = partial[0];
				}

				const ref = `${matched.provider}/${matched.id}`;
				if (ref === currentRef) {
					await sendTextReply(chatId, messageId, `Already using ${ref}.`);
					return;
				}

				const ok = await pi.setModel(matched);
				if (ok) {
					const nameSuffix = matched.name && matched.name !== matched.id ? ` (${matched.name})` : "";
					await sendTextReply(
						chatId,
						messageId,
						`Switched to ${ref}${nameSuffix}.`,
					);
				} else {
					await sendTextReply(
						chatId,
						messageId,
						`Cannot switch to ${ref} — no API key configured.`,
					);
				}
			} else {
				// Show model picker with inline keyboard
				const allModels = ctx.modelRegistry.getAvailable();
				const currentRef = ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : null;

				if (allModels.length === 0) {
					await sendTextReply(chatId, messageId, "No models available.");
					return;
				}

				// Build model entries for the picker
				const modelEntries = allModels.map((m) => ({
					ref: `${m.provider}/${m.id}`,
					provider: m.provider,
					id: m.id,
					name: m.name && m.name !== m.id ? m.name : undefined,
				}));

	

				// Store for callback handling
				pendingModelSelections.set(chatId, {
					models: modelEntries,
					messageId,
					page: 0,
				});

				// Render page 0
				await renderModelPicker(chatId, messageId, modelEntries, currentRef, 0);
			}
			return;
		}

		if (lower === "/help" || lower === "/start") {
			await sendTextReply(
				firstMessage.chat.id,
				firstMessage.message_id,
				`Send me a message and I will forward it to pi.\nTap the Menu button to browse pi's slash commands — type / to filter.\nRunnable from Telegram: /model, /status, /compact, /stop (or "stop"). Other menu entries are listed for reference; run them from the pi TUI.`,
			);
			if (config.allowedUserId === undefined && firstMessage.from) {
				config.allowedUserId = firstMessage.from.id;
				await writeConfig(config);
				updateStatus(ctx);
			}
			return;
		}

		if (lower.startsWith("/")) {
			const tgName = lower.slice(1).split(/[\s@]/)[0];
			const piName = activeReverseMap.get(tgName);
			if (piName) {
				// Commands that need an interactive TUI selector or dialog can't run
				// headless via pi.executeCommand — opening a picker on a Telegram-only
				// machine would just hang. Block them up-front.
				const TUI_ONLY = new Set([
					"settings", "login", "logout", "scoped-models",
					"fork", "clone", "tree", "resume", "import",
					"hotkeys", "changelog",
				]);
				if (TUI_ONLY.has(piName)) {
					await sendTextReply(
						firstMessage.chat.id,
						firstMessage.message_id,
						`/${piName} needs the pi TUI (interactive picker or dialog). Run it directly from pi.`,
					);
					return;
				}

				const argSpace = rawText.indexOf(" ");
				const cmdArgs = argSpace === -1 ? "" : rawText.slice(argSpace + 1).trim();
				const exec = (pi as { executeCommand?: (name: string, args?: string) => Promise<boolean> }).executeCommand;
				if (typeof exec !== "function") {
					await sendTextReply(
						firstMessage.chat.id,
						firstMessage.message_id,
						`/${piName}: this pi build doesn't support remote command dispatch. Upgrade pi or run the command from the TUI.`,
					);
					return;
				}
				try {
					const handled = await exec(piName, cmdArgs);
					if (!handled) {
						await sendTextReply(
							firstMessage.chat.id,
							firstMessage.message_id,
							`/${piName} is no longer registered.`,
						);
					}
					// On success: stay quiet — the command may produce a Telegram reply
					// itself (via pi.sendUserMessage), or simply mutate session state.
				} catch (error) {
					const message = error instanceof Error ? error.message : String(error);
					await sendTextReply(
						firstMessage.chat.id,
						firstMessage.message_id,
						`/${piName} failed: ${message}`,
					);
				}
				return;
			}
		}

		const historyTurns = preserveQueuedTurnsAsHistory ? queuedTelegramTurns.splice(0) : [];
		preserveQueuedTurnsAsHistory = false;
		const turn = await createTelegramTurn(messages, historyTurns);
		queuedTelegramTurns.push(turn);
		if (ctx.isIdle()) {
			startTypingLoop(ctx, turn.chatId);
			updateStatus(ctx);
			pi.sendUserMessage(turn.content);
		}
	}

	// Telegram inline keyboard limits: max 8 buttons per column, ~100 buttons total
	const MODEL_PAGE_SIZE = 5;

	function buildModelKeyboard(
		models: Array<{
			ref: string;
			provider: string;
			id: string;
			name?: string;
		}>,
		currentRef: string | null,
		page: number,
	): {
		text: string;
		keyboard: Array<Array<{ text: string; callback_data: string }>>;
	} {
		const totalPages = Math.ceil(models.length / MODEL_PAGE_SIZE);
		const start = page * MODEL_PAGE_SIZE;
		const end = Math.min(start + MODEL_PAGE_SIZE, models.length);
		const pageModels = models.slice(start, end);

		const headerLines = [
			`Models (${page + 1}/${totalPages})`,
			currentRef ? `Current: ${currentRef}` : "Current: none",
			"",
		];

		const keyboard: Array<Array<{ text: string; callback_data: string }>> = [];
		for (const m of pageModels) {
			const isCurrent = m.ref === currentRef;
			const label = m.name ? m.name : m.id;
			const prefix = isCurrent ? "✅ " : "  ";
			keyboard.push([{
				text: `${prefix}${label} (${m.provider})`,
				callback_data: `model:${m.ref}`,
			}]);
		}

		if (totalPages > 1) {
			const navRow: Array<{ text: string; callback_data: string }> = [];
			if (page > 0) navRow.push({ text: "⬅️ Prev", callback_data: `modelpage:${page - 1}` });
			if (page < totalPages - 1) navRow.push({ text: "Next ➡️", callback_data: `modelpage:${page + 1}` });
			if (navRow.length > 0) keyboard.push(navRow);
		}

		keyboard.push([{
			text: "❌ Cancel",
			callback_data: "model:cancel",
		}]);

		const text = headerLines.join("\n");
		return { text, keyboard };
	}

	async function renderModelPicker(
		chatId: number,
		_originalMessageId: number,
		models: Array<{
			ref: string;
			provider: string;
			id: string;
			name?: string;
		}>,
		currentRef: string | null,
		page: number,
		editMessageId?: number,
	): Promise<void> {
		const { text, keyboard } = buildModelKeyboard(models, currentRef, page);

		// Guard: ensure text is never empty (Telegram rejects empty text)
		if (!text || text.trim().length === 0) {
			console.error("[telepi-model] renderModelPicker: text is empty, models=", models.length, "page=", page);
			return;
		}

		try {
			if (editMessageId !== undefined) {
				await callTelegram<TelegramSentMessage>("editMessageText", {
					chat_id: chatId,
					message_id: editMessageId,
					text,
					reply_markup: { inline_keyboard: keyboard },
				});
			} else {
				const sent = await callTelegram<TelegramSentMessage>("sendMessage", {
					chat_id: chatId,
					text,
					reply_markup: { inline_keyboard: keyboard },
				});
				const pending = pendingModelSelections.get(chatId);
				if (pending) pending.messageId = sent.message_id;
			}
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			console.error("[telepi-model] renderModelPicker failed:", message, "text=", JSON.stringify(text.substring(0, 100)));
			await sendTextReply(chatId, _originalMessageId, `Failed to show model picker: ${message}`);
		}
	}

	async function replyAndCleanup(
		chatId: number,
		keyboardMessageId: number,
		resultText: string,
	): Promise<void> {
		try {
			await sendTextReply(chatId, keyboardMessageId, resultText);
		} catch (e) {
			console.error("[telepi-model] replyAndCleanup: reply failed:", e);
		}
		// Clean up the keyboard message
		try {
			await callTelegram<any>("deleteMessage", {
				chat_id: chatId,
				message_id: keyboardMessageId,
			});
		} catch {
			// keyboard might already be gone — ignore
		}
	}

	async function handleModelCallback(
		callbackQueryId: string,
		chatId: number,
		keyboardMessageId: number,
		data: string,
		ctx: ExtensionContext,
	): Promise<boolean> {


		try {
			await callTelegram<TelegramCallbackAnswer>("answerCallbackQuery", {
				callback_query_id: callbackQueryId,
			});
		} catch {
			// ignore
		}

		const pending = pendingModelSelections.get(chatId);
		if (!pending) return false;

		const currentRef = ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : null;

		if (data === "model:cancel") {
			pendingModelSelections.delete(chatId);
			await replyAndCleanup(chatId, keyboardMessageId, "Model picker cancelled.");
			return true;
		}

		if (data.startsWith("modelpage:")) {
			const newPage = parseInt(data.slice(10), 10);
			if (!isNaN(newPage) && newPage >= 0) {
				pending.page = newPage;
				try {
					await renderModelPicker(chatId, keyboardMessageId, pending.models, currentRef, newPage, keyboardMessageId);
				} catch (error) {
					const message = error instanceof Error ? error.message : String(error);
					await sendTextReply(chatId, keyboardMessageId, `Failed to render page: ${message}`);
				}
			}
			return true;
		}

		if (data.startsWith("model:")) {
			const ref = data.slice(6);
			const matched = pending.models.find((m) => m.ref === ref);
			if (!matched) return true;

			const model = ctx.modelRegistry.find(matched.provider, matched.id);
			if (!model) {
				pendingModelSelections.delete(chatId);
				await replyAndCleanup(chatId, keyboardMessageId, `Model ${ref} not found.`);
				return true;
			}

			if (ref === currentRef) {
				pendingModelSelections.delete(chatId);
				await replyAndCleanup(chatId, keyboardMessageId, `Already using ${ref}.`);
				return true;
			}

			const ok = await pi.setModel(model);
			if (ok) {
				pendingModelSelections.delete(chatId);
				const nameSuffix = model.name && model.name !== model.id ? ` (${model.name})` : "";
				await replyAndCleanup(chatId, keyboardMessageId, `Switched to ${ref}${nameSuffix}.`);
			} else {
				await replyAndCleanup(chatId, keyboardMessageId, `Cannot switch to ${ref} — no API key.`);
			}
			return true;
		}

		return false;
	}

	async function handleAuthorizedTelegramMessage(message: TelegramMessage, ctx: ExtensionContext): Promise<void> {
		if (message.media_group_id) {
			const key = `${message.chat.id}:${message.media_group_id}`;
			const existing = mediaGroups.get(key) ?? { messages: [] };
			existing.messages.push(message);
			if (existing.flushTimer) clearTimeout(existing.flushTimer);
			existing.flushTimer = setTimeout(() => {
				const state = mediaGroups.get(key);
				mediaGroups.delete(key);
				if (!state) return;
				void dispatchAuthorizedTelegramMessages(state.messages, ctx);
			}, TELEGRAM_MEDIA_GROUP_DEBOUNCE_MS);
			mediaGroups.set(key, existing);
			return;
		}

		await dispatchAuthorizedTelegramMessages([message], ctx);
	}

	async function handleUpdate(update: TelegramUpdate, ctx: ExtensionContext): Promise<void> {
		// Handle inline keyboard callbacks (model picker)
		if (update.callback_query) {
			const cq = update.callback_query;
			if (cq.from && cq.from.id !== config.allowedUserId) return;

			if (cq.data && cq.message) {
				const handled = await handleModelCallback(
					cq.id,
					cq.message.chat.id,
					cq.message.message_id,
					cq.data,
					ctx,
				);
				if (handled) return;
			}
			// Answer unhandled callbacks to clear loading state
			try {
				await callTelegram<TelegramCallbackAnswer>("answerCallbackQuery", {
					callback_query_id: cq.id,
					text: "Unknown action.",
					show_alert: false,
				});
			} catch {
				// ignore
			}
			return;
		}

		const message = update.message || update.edited_message;
		if (!message || message.chat.type !== "private" || !message.from || message.from.is_bot) return;

		if (config.allowedUserId === undefined) {
			config.allowedUserId = message.from.id;
			await writeConfig(config);
			updateStatus(ctx);
			await sendTextReply(message.chat.id, message.message_id, "Telegram bridge paired with this account.");
		}

		if (message.from.id !== config.allowedUserId) {
			await sendTextReply(message.chat.id, message.message_id, "This bot is not authorized for your account.");
			return;
		}

		await handleAuthorizedTelegramMessage(message, ctx);
	}

	async function pollLoop(ctx: ExtensionContext, signal: AbortSignal): Promise<void> {
		if (!config.botToken) return;

		try {
			await callTelegram("deleteWebhook", { drop_pending_updates: false }, { signal });
		} catch {
			// ignore
		}

		if (config.lastUpdateId === undefined) {
			try {
				const updates = await callTelegram<TelegramUpdate[]>("getUpdates", { offset: -1, limit: 1, timeout: 0 }, { signal });
				const last = updates.at(-1);
				if (last) {
					config.lastUpdateId = last.update_id;
					await writeConfig(config);
				}
			} catch {
				// ignore
			}
		}

		while (!signal.aborted) {
			try {
				const updates = await callTelegram<TelegramUpdate[]>(
					"getUpdates",
					{
						offset: config.lastUpdateId !== undefined ? config.lastUpdateId + 1 : undefined,
						limit: 10,
						timeout: 30,
						allowed_updates: ["message", "edited_message", "callback_query"],
					},
					{ signal },
				);
				for (const update of updates) {
					// If a previous update in this batch caused a session swap (e.g. /new
					// via pi.executeCommand), our session_shutdown handler aborts this
					// signal. Bail before touching the now-stale ctx — the new session's
					// own polling loop will fetch this update from offset+1.
					if (signal.aborted) break;
					config.lastUpdateId = update.update_id;
					await writeConfig(config);
					await handleUpdate(update, ctx);
				}
			} catch (error) {
				if (signal.aborted) return;
				if (error instanceof DOMException && error.name === "AbortError") return;
				const message = error instanceof Error ? error.message : String(error);
				updateStatus(ctx, message);
				await new Promise((resolve) => setTimeout(resolve, 3000));
				updateStatus(ctx);
			}
		}
	}

	async function startPolling(ctx: ExtensionContext): Promise<void> {
		if (!config.botToken || pollingPromise) return;
		pollingController = new AbortController();
		pollingPromise = pollLoop(ctx, pollingController.signal).finally(() => {
			pollingPromise = undefined;
			pollingController = undefined;
			updateStatus(ctx);
		});
		updateStatus(ctx);
		void syncTelegramCommands(ctx);
	}

	pi.registerTool({
		name: "telegram_attach",
		label: "Telegram Attach",
		description: "Queue one or more local files to be sent with the next Telegram reply.",
		promptSnippet: "Queue local files to be sent with the next Telegram reply.",
		promptGuidelines: [
			"When handling a [telegram] message and the user asked for a file or generated artifact, call telegram_attach with the local path instead of only mentioning the path in text.",
		],
		parameters: Type.Object({
			paths: Type.Array(Type.String({ description: "Local file path to attach" }), { minItems: 1, maxItems: MAX_ATTACHMENTS_PER_TURN }),
		}),
		async execute(_toolCallId, params) {
			if (!activeTelegramTurn) {
				throw new Error("telegram_attach can only be used while replying to an active Telegram turn");
			}
			const added: string[] = [];
			for (const inputPath of params.paths) {
				const stats = await stat(inputPath);
				if (!stats.isFile()) {
					throw new Error(`Not a file: ${inputPath}`);
				}
				if (activeTelegramTurn.queuedAttachments.length >= MAX_ATTACHMENTS_PER_TURN) {
					throw new Error(`Attachment limit reached (${MAX_ATTACHMENTS_PER_TURN})`);
				}
				activeTelegramTurn.queuedAttachments.push({ path: inputPath, fileName: basename(inputPath) });
				added.push(inputPath);
			}
			return {
				content: [{ type: "text", text: `Queued ${added.length} Telegram attachment(s).` }],
				details: { paths: added },
			};
		},
	});

	pi.registerCommand("telegram-setup", {
		description: "Configure Telegram bot token",
		handler: async (_args, ctx) => {
			await promptForConfig(ctx);
		},
	});

	pi.registerCommand("telegram-status", {
		description: "Show Telegram bridge status",
		handler: async (_args, ctx) => {
			const status = [
				`bot: ${config.botUsername ? `@${config.botUsername}` : "not configured"}`,
				`allowed user: ${config.allowedUserId ?? "not paired"}`,
				`polling: ${pollingPromise ? "running" : "stopped"}`,
				`active telegram turn: ${activeTelegramTurn ? "yes" : "no"}`,
				`queued telegram turns: ${queuedTelegramTurns.length}`,
			];
			ctx.ui.notify(status.join(" | "), "info");
		},
	});

	pi.registerCommand("telegram-connect", {
		description: "Start the Telegram bridge in this pi session",
		handler: async (_args, ctx) => {
			config = await readConfig();
			if (!config.botToken) {
				await promptForConfig(ctx);
				return;
			}
			await startPolling(ctx);
			updateStatus(ctx);
		},
	});

	pi.registerCommand("telegram-disconnect", {
		description: "Stop the Telegram bridge in this pi session",
		handler: async (_args, ctx) => {
			await stopPolling();
			updateStatus(ctx);
		},
	});

	pi.on("session_start", async (_event, ctx) => {
		config = await readConfig();
		await mkdir(TEMP_DIR, { recursive: true });
		if (config.botToken && !pollingPromise) { await startPolling(ctx); }
		updateStatus(ctx);
		void syncTelegramCommands(ctx);
	});

	pi.on("session_shutdown", async (_event, _ctx) => {
		queuedTelegramTurns = [];
		for (const state of mediaGroups.values()) {
			if (state.flushTimer) clearTimeout(state.flushTimer);
		}
		mediaGroups.clear();
		if (activeTelegramTurn) {
			await clearPreview(activeTelegramTurn.chatId);
		}
		activeTelegramTurn = undefined;
		currentAbort = undefined;
		preserveQueuedTurnsAsHistory = false;
		// Use the synchronous abort variant: this handler may run *on* the polling
		// task itself (when a Telegram-dispatched command triggers a session
		// replacement), so awaiting pollingPromise here would self-deadlock.
		abortPollingForShutdown();
	});

	pi.on("before_agent_start", async (event) => {
		const suffix = isTelegramPrompt(event.prompt)
			? `${SYSTEM_PROMPT_SUFFIX}\n- The current user message came from Telegram.`
			: SYSTEM_PROMPT_SUFFIX;
		return {
			systemPrompt: event.systemPrompt + suffix,
		};
	});

	pi.on("agent_start", async (_event, ctx) => {
		currentAbort = () => ctx.abort();
		if (!activeTelegramTurn && queuedTelegramTurns.length > 0) {
			const nextTurn = queuedTelegramTurns.shift();
			if (nextTurn) {
				activeTelegramTurn = { ...nextTurn };
				previewState = { mode: draftSupport === "unsupported" ? "message" : "draft", pendingText: "", lastSentText: "" };
				startTypingLoop(ctx);
			}
		}
		updateStatus(ctx);
	});

	pi.on("message_start", async (event, _ctx) => {
		if (!activeTelegramTurn || !isAssistantMessage(event.message)) return;
		if (previewState && (previewState.pendingText.trim().length > 0 || previewState.lastSentText.trim().length > 0)) {
			await finalizePreview(activeTelegramTurn.chatId);
		}
		previewState = { mode: draftSupport === "unsupported" ? "message" : "draft", pendingText: "", lastSentText: "" };
	});

	pi.on("message_update", async (event, _ctx) => {
		if (!activeTelegramTurn || !isAssistantMessage(event.message)) return;
		if (!previewState) {
			previewState = { mode: draftSupport === "unsupported" ? "message" : "draft", pendingText: "", lastSentText: "" };
		}
		previewState.pendingText = getMessageText(event.message);
		schedulePreviewFlush(activeTelegramTurn.chatId);
	});

	pi.on("agent_end", async (event, ctx) => {
		const turn = activeTelegramTurn;
		currentAbort = undefined;
		stopTypingLoop();
		activeTelegramTurn = undefined;
		updateStatus(ctx);
		if (!turn) return;

		const assistant = extractAssistantText(event.messages);
		if (assistant.stopReason === "aborted") {
			await clearPreview(turn.chatId);
			return;
		}
		if (assistant.stopReason === "error") {
			await clearPreview(turn.chatId);
			await sendTextReply(turn.chatId, turn.replyToMessageId, assistant.errorMessage || "Telegram bridge: pi failed while processing the request.");
			return;
		}

		const finalText = assistant.text;
		if (previewState) {
			previewState.pendingText = finalText ?? previewState.pendingText;
		}

		if (finalText && finalText.length <= MAX_MESSAGE_LENGTH) {
			const finalized = await finalizePreview(turn.chatId);
			if (!finalized && turn.queuedAttachments.length > 0 && !finalText) {
				await sendTextReply(turn.chatId, turn.replyToMessageId, "Attached requested file(s).");
			}
		} else {
			await clearPreview(turn.chatId);
			if (finalText) {
				await sendTextReply(turn.chatId, turn.replyToMessageId, finalText);
			} else if (turn.queuedAttachments.length > 0) {
				await sendTextReply(turn.chatId, turn.replyToMessageId, "Attached requested file(s).");
			}
		}

		await sendQueuedAttachments(turn);

		if (queuedTelegramTurns.length > 0 && !preserveQueuedTurnsAsHistory) {
			const nextTurn = queuedTelegramTurns[0];
			startTypingLoop(ctx, nextTurn.chatId);
			updateStatus(ctx);
			pi.sendUserMessage(nextTurn.content);
		}
	});
}
