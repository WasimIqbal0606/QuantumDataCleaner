    # OCR Extraction Tab
    with tabs[5]:
        st.subheader("OCR Time-Series Extraction")
        
        # Introduction
        st.markdown("""
        This tab shows the results of OCR extraction from images and PDFs. 
        Use the OCR Extract option in the sidebar to upload images or PDFs containing time-series data.
        """)
        
        # Check if OCR extraction was performed
        if "ocr_source_type" in st.session_state and st.session_state.ocr_source_type is not None:
            # Display extraction information
            st.markdown("### Extraction Information")
            
            # Display source type and quantum usage
            st.info(f"Source Type: {st.session_state.ocr_source_type.capitalize()}")
            if st.session_state.ocr_use_quantum:
                st.success("Used Quantum-Inspired Enhancement for better extraction quality")
            
            # Display metadata
            if st.session_state.ocr_metadata:
                st.markdown("### Extraction Metadata")
                
                metadata_expander = st.expander("View Extraction Details", expanded=True)
                with metadata_expander:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Execution Time", f"{st.session_state.ocr_metadata.get('execution_time_ms', 0):.1f} ms")
                        if "tables_found" in st.session_state.ocr_metadata:
                            st.metric("Tables Found", st.session_state.ocr_metadata["tables_found"])
                    
                    with col2:
                        if "date_column" in st.session_state.ocr_metadata and st.session_state.ocr_metadata["date_column"]:
                            st.metric("Date Column", st.session_state.ocr_metadata["date_column"])
                        if "value_column" in st.session_state.ocr_metadata and st.session_state.ocr_metadata["value_column"]:
                            st.metric("Value Column", st.session_state.ocr_metadata["value_column"])
                
                # Add success message if time series was found
                if st.session_state.ocr_metadata.get("time_series_found", False):
                    st.success("Time-series data successfully extracted!")
                else:
                    st.warning("No time-series data could be identified in the source.")
            
            # Display tabs for different views
            ocr_tabs = st.tabs(["Source Display", "Extracted Text", "Extracted Data"])
            
            # Source Display Tab
            with ocr_tabs[0]:
                if st.session_state.ocr_source_type == "image" and st.session_state.ocr_image is not None:
                    st.markdown("### Source Image")
                    st.image(st.session_state.ocr_image, caption="Source Image", use_column_width=True)
                elif st.session_state.ocr_source_type == "pdf":
                    st.markdown("### Source PDF")
                    st.info("PDF document was processed for OCR extraction.")
            
            # Extracted Text Tab
            with ocr_tabs[1]:
                if st.session_state.ocr_extracted_text:
                    st.markdown("### Extracted Text from OCR")
                    
                    # Format and display the text
                    st.code(st.session_state.ocr_extracted_text, language="text")
                    
                    # Add download button for extracted text
                    text_bytes = st.session_state.ocr_extracted_text.encode()
                    st.download_button(
                        label="Download Extracted Text",
                        data=text_bytes,
                        file_name=f"ocr_extracted_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                else:
                    st.info("No text was extracted from the source.")
            
            # Extracted Data Tab
            with ocr_tabs[2]:
                if st.session_state.ocr_extracted_df is not None:
                    st.markdown("### Extracted Time-Series Data")
                    
                    # Show dataframe preview
                    st.dataframe(st.session_state.ocr_extracted_df.head(20))
                    
                    # Show a plot if data is available
                    if len(st.session_state.ocr_extracted_df) > 0:
                        st.markdown("### Data Visualization")
                        
                        # Select a numeric column for plotting
                        numeric_cols = st.session_state.ocr_extracted_df.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            value_col = st.session_state.ocr_metadata.get("value_column", numeric_cols[0])
                            if value_col not in numeric_cols:
                                value_col = numeric_cols[0]
                            
                            # Create plot
                            fig = px.line(
                                st.session_state.ocr_extracted_df,
                                y=value_col,
                                title=f"Extracted Time Series: {value_col}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add download button for extracted data
                            csv_buffer = io.StringIO()
                            st.session_state.ocr_extracted_df.to_csv(csv_buffer)
                            st.download_button(
                                label="Download Extracted Data (CSV)",
                                data=csv_buffer.getvalue(),
                                file_name=f"ocr_extracted_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No numeric columns found for plotting.")
                    else:
                        st.warning("Extracted dataframe is empty.")
                else:
                    st.info("No structured data was extracted from the source.")
        else:
            st.info("No OCR extraction has been performed. Use the 'OCR Extract' option in the sidebar to upload images or PDFs.")