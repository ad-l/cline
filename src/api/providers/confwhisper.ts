import { Anthropic } from "@anthropic-ai/sdk"
import OpenAI, { AzureOpenAI } from "openai"
import { withRetry } from "../retry"
import { ApiHandlerOptions, azureOpenAiDefaultApiVersion, ModelInfo, openAiModelInfoSaneDefaults } from "@shared/api"
import { ApiHandler } from "../index"
import { convertToOpenAiMessages } from "../transform/openai-format"
import { ApiStream } from "../transform/stream"
import { convertToR1Format } from "../transform/r1-format"
import type { ChatCompletionReasoningEffort } from "openai/resources/chat/completions"
import * as ohttp from "attested-ohttp-client-ts"

export class ConfWhisperHandler implements ApiHandler {
	private options: ApiHandlerOptions
	private client: OpenAI
	private ohttpClient: Promise<ohttp.OhttpClient>

	constructor(options: ApiHandlerOptions) {
		this.options = options
		const clientBuilder = new ohttp.OhttpClientBuilder() //.withKmsCertPath("kms-cert.pem").withKmsUrl(this.options.confWhisperKMS || "");
		this.ohttpClient = clientBuilder.build()

		this.client = new OpenAI({
			baseURL: this.options.confWhisperUrl,
			apiKey: this.options.confWhisperKey,
			defaultHeaders: this.options.confWhisperHeaders,
			//	fetch: this.ohttp_fetch,
		})
	}

	/*
	async ohttp_fetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
		let req0 = new Request(input, init);
		let cli = await this.ohttpClient;
		const ctx = await cli.encapsulateRequest(req0);
		let req = ctx.request.request();
		return fetch(req).then(async (res) => {
			let stream = await ctx.decapsulateResponseChunked(res);
			let res1 = new Response(stream, {status:res.status, statusText:res.statusText, headers:res.headers});
			return res;
		});
	}

		*/

	@withRetry()
	async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
		const modelId = this.options.openAiModelId ?? ""
		const isDeepseekReasoner = modelId.includes("deepseek-reasoner")
		const isR1FormatRequired = this.options.openAiModelInfo?.isR1FormatRequired ?? false
		const isReasoningModelFamily = modelId.includes("o1") || modelId.includes("o3") || modelId.includes("o4")

		let openAiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [
			{ role: "system", content: systemPrompt },
			...convertToOpenAiMessages(messages),
		]
		let temperature: number | undefined = this.options.openAiModelInfo?.temperature ?? openAiModelInfoSaneDefaults.temperature
		let reasoningEffort: ChatCompletionReasoningEffort | undefined = undefined
		let maxTokens: number | undefined

		if (this.options.openAiModelInfo?.maxTokens && this.options.openAiModelInfo.maxTokens > 0) {
			maxTokens = Number(this.options.openAiModelInfo.maxTokens)
		} else {
			maxTokens = undefined
		}

		if (isDeepseekReasoner || isR1FormatRequired) {
			openAiMessages = convertToR1Format([{ role: "user", content: systemPrompt }, ...messages])
		}

		if (isReasoningModelFamily) {
			openAiMessages = [{ role: "developer", content: systemPrompt }, ...convertToOpenAiMessages(messages)]
			temperature = undefined // does not support temperature
			reasoningEffort = (this.options.o3MiniReasoningEffort as ChatCompletionReasoningEffort) || "medium"
		}

		const stream = await this.client.chat.completions.create({
			model: modelId,
			messages: openAiMessages,
			temperature,
			max_tokens: maxTokens,
			reasoning_effort: reasoningEffort,
			stream: true,
			stream_options: { include_usage: true },
		})
		for await (const chunk of stream) {
			const delta = chunk.choices[0]?.delta
			if (delta?.content) {
				yield {
					type: "text",
					text: delta.content,
				}
			}

			if (delta && "reasoning_content" in delta && delta.reasoning_content) {
				yield {
					type: "reasoning",
					reasoning: (delta.reasoning_content as string | undefined) || "",
				}
			}

			if (chunk.usage) {
				yield {
					type: "usage",
					inputTokens: chunk.usage.prompt_tokens || 0,
					outputTokens: chunk.usage.completion_tokens || 0,
				}
			}
		}
	}

	getModel(): { id: string; info: ModelInfo } {
		return {
			id: this.options.openAiModelId ?? "deepseek-ai/DeepSeek-R1",
			info: this.options.openAiModelInfo ?? openAiModelInfoSaneDefaults,
		}
	}
}
