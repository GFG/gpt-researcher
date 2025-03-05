# researcher.py - Core research orchestration for GPT-Researcher
#
# This file defines the ResearchConductor class which manages the entire research process.
# It handles different research sources (web, local documents, vector stores) and
# coordinates the gathering and processing of information.
#
# Key workflow:
# 1. Initialize with a researcher instance (GPTResearcher)
# 2. Plan research by generating sub-queries
# 3. Gather information from appropriate sources based on configuration
# 4. Process and filter the gathered information
# 5. Return a consolidated research context for report generation

import asyncio
import random
import logging
import os
from ..actions.utils import stream_output
from ..actions.query_processing import plan_research_outline, get_search_results
from ..document import DocumentLoader, OnlineDocumentLoader, LangChainDocumentLoader
from ..utils.enum import ReportSource
from ..utils.logging_config import get_json_handler
from typing import Optional, List, Dict, Any


class ResearchConductor:
    """Manages and coordinates the research process."""

    def __init__(self, researcher):
        # Store reference to the parent GPTResearcher instance
        self.researcher = researcher
        self.logger = logging.getLogger('research')
        self.json_handler = get_json_handler()

    async def plan_research(self, query, query_domains=None):
        """
        Plan the research by generating sub-queries based on the main query.
        
        This method:
        1. Performs initial web search to understand the topic
        2. Uses LLM to generate a research outline with sub-queries
        3. Returns a list of sub-queries to investigate
        """
        self.logger.info(f"Planning research for query: {query}")
        if query_domains:
            self.logger.info(f"Query domains: {query_domains}")
        
        # Notify user that we're starting the research planning
        await stream_output(
            "logs",
            "planning_research",
            f"🌐 Browsing the web to learn more about the task: {query}...",
            self.researcher.websocket,
        )

        # Get initial search results to understand the topic
        search_results = await get_search_results(query, self.researcher.retrievers[0], query_domains)
        self.logger.info(f"Initial search results obtained: {len(search_results)} results")

        await stream_output(
            "logs",
            "planning_research",
            f"🤔 Planning the research strategy and subtasks...",
            self.researcher.websocket,
        )

        # Generate a research outline with sub-queries using LLM
        outline = await plan_research_outline(
            query=query,
            search_results=search_results,
            agent_role_prompt=self.researcher.role,
            cfg=self.researcher.cfg,
            parent_query=self.researcher.parent_query,
            report_type=self.researcher.report_type,
            cost_callback=self.researcher.add_costs,
        )
        self.logger.info(f"Research outline planned: {outline}")
        return outline

    async def conduct_research(self):
        """
        Main research orchestration method.
        
        This method:
        1. Determines which research source to use (web, local, vector store, etc.)
        2. Calls the appropriate methods to gather information
        3. Processes and curates the gathered information
        4. Returns the consolidated research context
        """
        if self.json_handler:
            self.json_handler.update_content("query", self.researcher.query)
        
        self.logger.info(f"Starting research for query: {self.researcher.query}")
        
        # Reset visited_urls and source_urls at the start of each research task
        self.researcher.visited_urls.clear()
        research_data = []

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "starting_research",
                f"🔍 Starting the research task for '{self.researcher.query}'...",
                self.researcher.websocket,
            )
            await stream_output(
                "logs",
                "agent_generated",
                self.researcher.agent,
                self.researcher.websocket
            )

        # RESEARCH SOURCE SELECTION
        # The system supports multiple research sources and selects the appropriate one
        # based on configuration. Each source has its own data gathering approach.
        
        # 1. SPECIFIC SOURCE URLS: Use provided URLs as research sources
        if self.researcher.source_urls:
            self.logger.info("Using provided source URLs")
            research_data = await self._get_context_by_urls(self.researcher.source_urls)
            if research_data and len(research_data) == 0 and self.researcher.verbose:
                await stream_output(
                    "logs",
                    "answering_from_memory",
                    f"🧐 I was unable to find relevant context in the provided sources...",
                    self.researcher.websocket,
                )
            # Optionally complement with web search if configured
            if self.researcher.complement_source_urls:
                self.logger.info("Complementing with web search")
                additional_research = await self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains)
                research_data += ' '.join(additional_research)

        # 2. WEB SEARCH: Use web search as the primary research source
        elif self.researcher.report_source == ReportSource.Web.value:
            self.logger.info("Using web search")
            research_data = await self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains)
            
        # 3. LOCAL DOCUMENTS: Use local document search
        # This loads documents from a local path and optionally adds them to the vector store
        elif self.researcher.report_source == ReportSource.Local.value:
            self.logger.info("Using local search")
            document_data = await DocumentLoader(self.researcher.cfg.doc_path).load()
            self.logger.info(f"Loaded {len(document_data)} documents")
            
            # If vector store is available, load documents into it for searching
            if self.researcher.vector_store:
                self.researcher.vector_store.load(document_data)

            research_data = await self._get_context_by_web_search(self.researcher.query, document_data, self.researcher.query_domains)

        # 4. HYBRID SEARCH: Combine vector db (internal knowledge base) and web search
        # This approach uses both document sources and web sources
        elif self.researcher.report_source == ReportSource.Hybrid.value:
            # Get context from both document sources and web sources
            docs_context = await self._get_context_by_vectorstore(self.researcher.query, self.researcher.vector_store_filter)
            web_context = await self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains)
            # research_data = f"Context from the internal knowledge base: {docs_context}\n\nContext from the web: {web_context}"
            research_data = docs_context + web_context

        # 5. AZURE STORAGE: Use documents from Azure blob storage
        elif self.researcher.report_source == ReportSource.Azure.value:
            from ..document.azure_document_loader import AzureDocumentLoader
            azure_loader = AzureDocumentLoader(
                container_name=os.getenv("AZURE_CONTAINER_NAME"),
                connection_string=os.getenv("AZURE_CONNECTION_STRING")
            )
            azure_files = await azure_loader.load()
            document_data = await DocumentLoader(azure_files).load()  # Reuse existing loader
            research_data = await self._get_context_by_web_search(self.researcher.query, document_data)

        # 6. LANGCHAIN DOCUMENTS: Use provided Langchain documents
        elif self.researcher.report_source == ReportSource.LangChainDocuments.value:
            langchain_documents_data = await LangChainDocumentLoader(
                self.researcher.documents
            ).load()
            
            # If vector store is available, load documents into it
            if self.researcher.vector_store:
                self.researcher.vector_store.load(langchain_documents_data)
                
            research_data = await self._get_context_by_web_search(
                self.researcher.query, langchain_documents_data, self.researcher.query_domains
            )

        # 7. VECTOR STORE: Use the provided vector store directly
        # This is the most direct use of the vector store - searching it without loading new documents
        elif self.researcher.report_source == ReportSource.LangChainVectorStore.value:
            research_data = await self._get_context_by_vectorstore(self.researcher.query, self.researcher.vector_store_filter)

        # FINALIZE RESEARCH
        # Store the gathered research data and optionally curate it
        self.researcher.context = research_data
        if self.researcher.cfg.curate_sources:
            self.logger.info("Curating sources")
            self.researcher.context = await self.researcher.source_curator.curate_sources(research_data)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "research_step_finalized",
                f"Finalized research step.\n💸 Total Research Costs: ${self.researcher.get_costs()}",
                self.researcher.websocket,
            )
            if self.json_handler:
                self.json_handler.update_content("costs", self.researcher.get_costs())
                self.json_handler.update_content("context", self.researcher.context)

        self.logger.info(f"Research completed. Context size: {len(str(self.researcher.context))}")
        
        return self.researcher.context

    async def _get_context_by_urls(self, urls):
        """
        Scrapes content from provided URLs and adds it to the research context.
        
        This method:
        1. Filters out already visited URLs
        2. Scrapes content from new URLs
        3. Optionally loads content into vector store
        4. Retrieves relevant content based on the query
        """
        self.logger.info(f"Getting context from URLs: {urls}")
        
        new_search_urls = await self._get_new_urls(urls)
        self.logger.info(f"New URLs to process: {new_search_urls}")

        # Scrape content from the URLs
        scraped_content = await self.researcher.scraper_manager.browse_urls(new_search_urls)
        self.logger.info(f"Scraped content from {len(scraped_content)} URLs")

        # If vector store is available, load scraped content into it
        # This allows the content to be searched in future queries
        if self.researcher.vector_store:
            self.logger.info("Loading content into vector store")
            self.researcher.vector_store.load(scraped_content)

        # Get relevant content based on the query
        context = await self.researcher.context_manager.get_similar_content_by_query(
            self.researcher.query, scraped_content
        )
        return context

    async def _get_context_by_vectorstore(self, query, filter: Optional[dict] = None):
        """
        Searches the vector store for relevant content based on the query.
        
        This method:
        1. Plans research by generating sub-queries
        2. Searches the vector store for each sub-query
        3. Returns the combined search results
        
        This is the primary method that uses the vector store for retrieval.
        """
        self.logger.info(f"Starting vectorstore search for query: {query}")
        context = []
        # Generate Sub-Queries including original query
        sub_queries = await self.plan_research(query)
        # If this is not part of a sub researcher, add original query to research for better results
        if self.researcher.report_type != "subtopic_report":
            sub_queries.append(query)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "subqueries",
                f"🗂️  I will conduct my research based on the following queries: {sub_queries}...",
                self.researcher.websocket,
                True,
                sub_queries,
            )

        # Using asyncio.gather to process the sub_queries asynchronously
        # Each sub-query is searched in the vector store
        context = await asyncio.gather(
            *[
                self._process_sub_query_with_vectorstore(sub_query, filter)
                for sub_query in sub_queries
            ]
        )
        return context

    async def _get_context_by_web_search(self, query, scraped_data: list | None = None, query_domains: list | None = None):
        """
        Generates the context for the research task by searching the query and scraping the results
        Returns:
            context: List of context
        """
        self.logger.info(f"Starting web search for query: {query}")
        
        if scraped_data is None:
            scraped_data = []
        if query_domains is None:
            query_domains = []

        # Generate Sub-Queries including original query
        sub_queries = await self.plan_research(query, query_domains)
        self.logger.info(f"Generated sub-queries: {sub_queries}")
        
        # If this is not part of a sub researcher, add original query to research for better results
        if self.researcher.report_type != "subtopic_report":
            sub_queries.append(query)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "subqueries",
                f"🗂️ I will conduct my research based on the following queries: {sub_queries}...",
                self.researcher.websocket,
                True,
                sub_queries,
            )

        # Using asyncio.gather to process the sub_queries asynchronously
        try:
            context = await asyncio.gather(
                *[
                    self._process_sub_query(sub_query, scraped_data, query_domains)
                    for sub_query in sub_queries
                ]
            )
            self.logger.info(f"Gathered context from {len(context)} sub-queries")
            # Filter out empty results and join the context
            context = [c for c in context if c]
            if context:
                combined_context = " ".join(context)
                self.logger.info(f"Combined context size: {len(combined_context)}")
                return combined_context
            return []
        except Exception as e:
            self.logger.error(f"Error during web search: {e}", exc_info=True)
            return []

    async def _process_sub_query(self, sub_query: str, scraped_data: list = [], query_domains: list = []):
        """Takes in a sub query and scrapes urls based on it and gathers context."""
        if self.json_handler:
            self.json_handler.log_event("sub_query", {
                "query": sub_query,
                "scraped_data_size": len(scraped_data)
            })
        
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "running_subquery_research",
                f"\n🔍 Running research for '{sub_query}'...",
                self.researcher.websocket,
            )

        try:
            # If no scraped data is provided, scrape new data
            if not scraped_data:
                scraped_data = await self._scrape_data_by_urls(sub_query, query_domains)
                self.logger.info(f"Scraped data size: {len(scraped_data)}")

            # Get relevant content based on the sub-query
            content = await self.researcher.context_manager.get_similar_content_by_query(sub_query, scraped_data)
            self.logger.info(f"Content found for sub-query: {len(str(content)) if content else 0} chars")

            # Log the content if available
            if content and self.researcher.verbose:
                # await stream_output(
                #     "logs", "subquery_context_window", f"📃 {content}", self.researcher.websocket
                # )
                pass
            elif self.researcher.verbose:
                await stream_output(
                    "logs",
                    "subquery_context_not_found",
                    f"🤷 No content found for '{sub_query}'...",
                    self.researcher.websocket,
                )
            if content:
                if self.json_handler:
                    self.json_handler.log_event("content_found", {
                        "sub_query": sub_query,
                        "content_size": len(content)
                    })
            return content
        except Exception as e:
            self.logger.error(f"Error processing sub-query {sub_query}: {e}", exc_info=True)
            return ""

    async def _process_sub_query_with_vectorstore(self, sub_query: str, filter: dict | None = None):
        """Takes in a sub query and gathers context from the user provided vector store

        Args:
            sub_query (str): The sub-query generated from the original query

        Returns:
            str: The context gathered from search
        """
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "running_subquery_with_vectorstore_research",
                f"\n🔍 Running research for '{sub_query}'...",
                self.researcher.websocket,
            )

        # Search the vector store for content relevant to the sub-query
        # This calls a method in the context_manager that performs the actual search
        content = await self.researcher.context_manager.get_similar_content_by_query_with_vectorstore(sub_query, filter)

        # Log the content if available
        if content and self.researcher.verbose:
            # await stream_output(
            #     "logs", "subquery_context_window", f"📃 {content}", self.researcher.websocket
            # )
            pass
        elif self.researcher.verbose:
            await stream_output(
                "logs",
                "subquery_context_not_found",
                f"🤷 No content found for '{sub_query}'...",
                self.researcher.websocket,
            )
        return content

    async def _get_new_urls(self, url_set_input):
        """Gets the new urls from the given url set.
        Args: url_set_input (set[str]): The url set to get the new urls from
        Returns: list[str]: The new urls from the given url set
        """

        new_urls = []
        for url in url_set_input:
            if url not in self.researcher.visited_urls:
                self.researcher.visited_urls.add(url)
                new_urls.append(url)
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "added_source_url",
                        f"- Added source url to research: {url}\n",
                        self.researcher.websocket,
                        True,
                        url,
                    )

        return new_urls

    async def _search_relevant_source_urls(self, query, query_domains: list | None = None):
        new_search_urls = []
        if query_domains is None:
            query_domains = []

        # Iterate through all retrievers
        for retriever_class in self.researcher.retrievers:
            # Instantiate the retriever with the sub-query, handle incompatible parameters
            try:
                if query_domains:
                    try:
                        retriever = retriever_class(query, query_domains=query_domains)
                    except TypeError:
                        # If query_domains isn't supported, initialize without it
                        retriever = retriever_class(query)
                else:
                    retriever = retriever_class(query)

                # Perform the search using the current retriever
                search_results = await asyncio.to_thread(
                    retriever.search, max_results=self.researcher.cfg.max_search_results_per_query
                )

                # Collect new URLs from search results
                search_urls = [url.get("href") for url in search_results]
                new_search_urls.extend(search_urls)
            except Exception as e:
                self.logger.error(f"Error with retriever {retriever_class.__name__}: {e}")
                continue

        # Get unique URLs
        new_search_urls = await self._get_new_urls(new_search_urls)
        random.shuffle(new_search_urls)

        return new_search_urls

    async def _scrape_data_by_urls(self, sub_query, query_domains: list | None = None):
        """
        Runs a sub-query across multiple retrievers and scrapes the resulting URLs.

        Args:
            sub_query (str): The sub-query to search for.

        Returns:
            list: A list of scraped content results.
        """
        if query_domains is None:
            query_domains = []

        new_search_urls = await self._search_relevant_source_urls(sub_query, query_domains)

        # Log the research process if verbose mode is on
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "researching",
                f"🤔 Researching for relevant information across multiple sources...\n",
                self.researcher.websocket,
            )

        # Scrape the new URLs
        scraped_content = await self.researcher.scraper_manager.browse_urls(new_search_urls)

        if self.researcher.vector_store:
            self.researcher.vector_store.load(scraped_content)

        return scraped_content
