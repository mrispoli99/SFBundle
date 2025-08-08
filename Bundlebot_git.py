# app.py
import os, time, json, textwrap
from typing import List, Optional, Dict
import requests
import streamlit as st
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

try:
    import streamlit as st  # already imported
    for k in ("SHOPIFY_STORE","SHOPIFY_TOKEN","GEMINI_API_KEY","API_VERSION","ADMIN_PASSWORD"):
        if k in st.secrets and not os.environ.get(k):
            os.environ[k] = str(st.secrets[k])
except Exception:
    pass
    
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.title("🔒 Summer Fridays Bundle Builder Login")
    pwd = st.text_input("SFBundle2025", type="password")
    if st.button("Enter"):
        if pwd and ADMIN_PASSWORD and pwd == ADMIN_PASSWORD:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Nope.")
    st.stop()


# =========================
# Config & setup
# =========================
load_dotenv()
SHOP = os.environ["SHOPIFY_STORE"]
TOKEN = os.environ["SHOPIFY_TOKEN"]
API_VERSION = os.environ.get("API_VERSION", "2025-07")
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

ADMIN_URL = f"https://{SHOP}/admin/api/{API_VERSION}/graphql.json"
HEADERS = {"Content-Type": "application/json", "X-Shopify-Access-Token": TOKEN}

genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"

st.set_page_config(page_title="Shopify Bundle Builder (Fixed)", page_icon="🧩", layout="wide")
st.title("🧩 Summer Fridays Bundle Builder")

# =========================
# Models (no JSON-Schema constraints)
# =========================
class ComponentDraft(BaseModel):
    variant_id: str
    quantity: int = 1  # clamp >=1 later

class BundleDraft(BaseModel):
    title: str
    description_html: str
    price: float
    compare_at_price: Optional[float] = None
    components: List[ComponentDraft]

# =========================
# Gemini helpers (raw JSON; validate with Pydantic)
# =========================
def gemini_propose_bundle_draft(context: Dict) -> BundleDraft:
    """
    Ask Gemini to return raw JSON (no response_schema). Validate with Pydantic.
    """
    sys = textwrap.dedent("""
        You are a Shopify merchandising assistant.
        Return ONLY a single JSON object (no prose) with this schema:
        {
          "title": string,                  // <= 60 chars, specific
          "description_html": string,       // brief marketing copy (valid HTML; can include <ul><li>)
          "price": number,                  // non-negative; .00 or .99 if reasonable
          "compare_at_price": number|null,  // optional; >= price if discounting; else null
          "components": [                   // use provided variant_ids; don't invent
            {"variant_id": string, "quantity": number}
          ]
        }
        Rules:
        - Use the exact variant_ids from context; default quantity = 1 unless user asked otherwise.
        - Do not invent products/SKUs/IDs.
    """).strip()

    model = genai.GenerativeModel(
        MODEL_NAME,
        generation_config={"response_mime_type": "application/json"}
    )

    prompt = (
        "Selected variants and notes:\n"
        + json.dumps(context, indent=2)
        + "\nReturn ONLY the JSON object."
    )

    resp = model.generate_content([sys, prompt])
    data = json.loads(resp.text)
    return BundleDraft(**data)

def gemini_suggest_companions(anchor: Dict, candidates: List[Dict], k: int = 2) -> List[str]:
    """
    Ask Gemini to choose k companion variant_ids from provided candidates (by id),
    based on complementarity, vendor/productType/tags/title, and price sanity.
    Returns a list of variant_id strings (subset of the provided candidates).
    """
    sys = """
    You are a Shopify merchandising assistant. Choose complementary products
    to form a sensible fixed bundle around the anchor product.

    Return ONLY a JSON object like:
    {"variant_ids": ["gid://shopify/ProductVariant/123", "gid://shopify/ProductVariant/456"]}

    Rules:
    - Use ONLY variant_ids that appear in candidates.
    - Pick distinct variant_ids; exclude the anchor's variant_id.
    - Prefer complements (e.g., cleanser + toner + moisturizer); same vendor when sensible.
    - Use productType/tags/title to infer fit.
    - Keep price mix reasonable for a bundle.
    """
    model = genai.GenerativeModel(
        MODEL_NAME,
        generation_config={"response_mime_type": "application/json"}
    )

    anchor_compact = {
        "variant_id": anchor["variant"]["id"],
        "variant_title": anchor["variant"]["title"],
        "product_title": anchor["product"]["title"],
        "vendor": anchor["product"].get("vendor"),
        "productType": anchor["product"].get("productType"),
        "tags": anchor["product"].get("tags"),
        "price": anchor["variant"].get("price"),
    }

    # Keep candidates compact + bounded
    cand_compact = []
    for c in candidates[:200]:
        cand_compact.append({
            "variant_id": c["variant"]["id"],
            "variant_title": c["variant"]["title"],
            "product_title": c["product"]["title"],
            "vendor": c["product"].get("vendor"),
            "productType": c["product"].get("productType"),
            "tags": c["product"].get("tags"),
            "price": c["variant"].get("price"),
        })

    prompt = (
        f"anchor:\n{json.dumps(anchor_compact, indent=2)}\n\n"
        f"candidates:\n{json.dumps(cand_compact, indent=2)}\n\n"
        f"Pick exactly {k} companion variant_ids from the candidates (exclude the anchor)."
    )
    resp = model.generate_content([sys, prompt])
    try:
        data = json.loads(resp.text)
        variant_ids = data.get("variant_ids") or []
        return [str(v) for v in variant_ids][:k]
    except Exception:
        return []

# =========================
# Shopify GraphQL helpers (retries + pagination)
# =========================
def shopify_graphql(query: str, variables: dict = None, max_retries: int = 6):
    payload = {"query": query, "variables": variables or {}}
    backoff = 1.0
    for _ in range(max_retries):
        r = requests.post(ADMIN_URL, headers=HEADERS, json=payload, timeout=90)
        if r.status_code == 429:
            time.sleep(backoff); backoff = min(backoff * 2, 8); continue
        r.raise_for_status()
        data = r.json()
        if "errors" in data:
            if any("throttle" in (e.get("message","") + str(e)).lower() for e in data["errors"]):
                time.sleep(backoff); backoff = min(backoff * 2, 8); continue
            raise RuntimeError(data["errors"])
        return data["data"]
    raise RuntimeError("Shopify GraphQL: exceeded retry budget (throttled?)")

def _fetch_products_page(after: Optional[str] = None, first: int = 250):
    # fetch richer product fields to help AI suggestion
    query = """
    query($first:Int!, $after:String){
      products(first:$first, after:$after, sortKey:TITLE) {
        edges {
          cursor
          node {
            id
            title
            vendor
            productType
            tags
            options { id name values }
            variants(first:250) {
              edges {
                node {
                  id
                  title
                  price
                  selectedOptions { name value }
                }
              }
              pageInfo { hasNextPage endCursor }
            }
          }
        }
        pageInfo { hasNextPage endCursor }
      }
    }
    """
    return shopify_graphql(query, {"first": first, "after": after})["products"]

def _fetch_more_variants(product_id: str, after: Optional[str]):
    query = """
    query($id:ID!, $after:String){
      node(id:$id){
        ... on Product {
          id
          variants(first:250, after:$after){
            edges {
              node {
                id
                title
                price
                selectedOptions { name value }
              }
            }
            pageInfo { hasNextPage endCursor }
          }
        }
      }
    }
    """
    return shopify_graphql(query, {"id": product_id, "after": after})["node"]["variants"]

def fetch_all_products_variants() -> List[dict]:
    """Return product nodes with ALL variants paged in."""
    products: List[dict] = []
    after_products = None
    while True:
        page = _fetch_products_page(after_products, first=250)
        for edge in page["edges"]:
            p = edge["node"]
            v_page = p["variants"]
            while v_page["pageInfo"]["hasNextPage"]:
                v_page = _fetch_more_variants(p["id"], v_page["pageInfo"]["endCursor"])
                p["variants"]["edges"].extend(v_page["edges"])
            products.append(p)
        if not page["pageInfo"]["hasNextPage"]:
            break
        after_products = page["pageInfo"]["endCursor"]
        time.sleep(0.3)  # be polite to the API
    return products

def build_component_inputs(selected_variant_ids: List[str], product_index: Dict[str, dict]):
    """Build ProductBundleComponentInput[] mapping variant selectedOptions -> product.options."""
    components = []
    for vgid in selected_variant_ids:
        prod = product_index[vgid]["product"]
        variant = product_index[vgid]["variant"]
        option_id_by_name = {opt["name"]: opt["id"] for opt in prod["options"]}
        optionSelections = []
        for so in variant["selectedOptions"]:
            name, value = so["name"], so["value"]
            if name in option_id_by_name:
                optionSelections.append({
                    "componentOptionId": option_id_by_name[name],
                    "name": name,
                    "values": [value],
                })
        components.append({
            "productId": prod["id"],
            "quantity": 1,
            "optionSelections": optionSelections,
        })
    return components

def product_bundle_create(title: str, components: List[dict]) -> str:
    mutation = """
    mutation($input: ProductBundleCreateInput!) {
      productBundleCreate(input: $input) {
        productBundleOperation { id status }
        userErrors { field message }
      }
    }
    """
    data = shopify_graphql(mutation, {"input": {"title": title, "components": components}})
    res = data["productBundleCreate"]
    if res["userErrors"]:
        raise RuntimeError(res["userErrors"])
    return res["productBundleOperation"]["id"]

def product_operation_poll(op_id: str, timeout_s=120) -> str:
    query = """
    query($id: ID!) {
      productOperation(id: $id) {
        __typename
        ... on ProductBundleOperation {
          status
          product { id }
          userErrors { message code }
        }
      }
    }
    """
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        op = shopify_graphql(query, {"id": op_id})["productOperation"]
        status = op.get("status")
        if status == "COMPLETE":
            prod = op.get("product")
            if not prod:
                raise RuntimeError("Bundle op COMPLETE but no product returned.")
            return prod["id"]
        elif status in ("ACTIVE", "CREATED"):
            time.sleep(2)
        else:
            raise RuntimeError(f"Bundle operation failed: {op}")
    raise TimeoutError("Timed out waiting for bundle creation to complete.")

def product_update(product_id: str, description_html: str, status: str = "ACTIVE"):
    mutation = """
    mutation($product: ProductUpdateInput!) {
      productUpdate(product: $product) {
        product { id status }
        userErrors { field message }
      }
    }
    """
    res = shopify_graphql(mutation, {"product": {"id": product_id, "status": status, "descriptionHtml": description_html}})
    if res["productUpdate"]["userErrors"]:
        raise RuntimeError(res["productUpdate"]["userErrors"])

def get_default_variant_id(product_id: str) -> str:
    query = """
    query($id: ID!){
      node(id:$id){
        ... on Product {
          variants(first:1){ nodes { id } }
        }
      }
    }
    """
    nodes = shopify_graphql(query, {"id": product_id})["node"]["variants"]["nodes"]
    if not nodes:
        raise RuntimeError("No variants on created bundle product.")
    return nodes[0]["id"]

def set_price_on_variant(product_id: str, variant_id: str, price: float, compare_at_price: Optional[float]):
    """
    productVariantsBulkUpdate requires productId.
    """
    mutation = """
    mutation($productId: ID!, $variants:[ProductVariantsBulkInput!]!) {
      productVariantsBulkUpdate(productId: $productId, variants: $variants) {
        productVariants { id }
        userErrors { field message }
      }
    }
    """
    v = {"id": variant_id, "price": f"{round(float(price),2):.2f}"}
    cap = None if compare_at_price in (None, "") else float(compare_at_price)
    if cap is not None and cap > 0:
        v["compareAtPrice"] = f"{round(cap,2):.2f}"
    res = shopify_graphql(mutation, {"productId": product_id, "variants": [v]})["productVariantsBulkUpdate"]
    if res["userErrors"]:
        raise RuntimeError(res["userErrors"])

def publish_to_current_channel(product_id: str):
    mutation = """
    mutation($id: ID!){
      publishablePublishToCurrentChannel(id:$id){
        publishable { ... on Product { id } }
        userErrors { field message }
      }
    }
    """
    res = shopify_graphql(mutation, {"id": product_id})["publishablePublishToCurrentChannel"]
    if res["userErrors"]:
        raise RuntimeError(res["userErrors"])

# Robust rerun helper
def _rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# =========================
# Load catalog once
# =========================
if "catalog" not in st.session_state:
    with st.spinner("Loading products & variants (first load may take a minute)…"):
        products = fetch_all_products_variants()
        # Build variant index & display labels
        var_index = {}
        choices = []
        for p in products:
            for ve in p["variants"]["edges"]:
                v = ve["node"]
                label = f'{p["title"]} — {v["title"]}  ({v["id"].split("/")[-1]})'
                choices.append((label, v["id"]))
                var_index[v["id"]] = {"product": p, "variant": v}
        st.session_state.catalog = {"products": products, "var_index": var_index, "choices": choices}

catalog = st.session_state.catalog
labels = [label for label, _ in catalog["choices"]]
id_by_label = {label: vid for label, vid in catalog["choices"]}

# Session state
if "picked_variant_ids" not in st.session_state:
    st.session_state.picked_variant_ids = []
if "search_nonce" not in st.session_state:
    st.session_state.search_nonce = 0
if "last_suggested_vids" not in st.session_state:
    st.session_state.last_suggested_vids = []

# -------------------------
# Candidate pool for AI suggestions
# -------------------------
def _candidate_pool_for_anchor(anchor_vid: str, var_index: Dict[str, Dict]) -> List[Dict]:
    """Create a candidate list: same vendor first, then same productType, then everything else."""
    anchor = var_index[anchor_vid]
    avendor = (anchor["product"].get("vendor") or "").lower()
    atype = (anchor["product"].get("productType") or "").lower()

    same_vendor, same_type, others = [], [], []
    for vid, entry in var_index.items():
        if vid == anchor_vid:
            continue
        vendor = (entry["product"].get("vendor") or "").lower()
        ptype = (entry["product"].get("productType") or "").lower()
        if avendor and vendor == avendor:
            same_vendor.append(entry)
        elif atype and ptype == atype:
            same_type.append(entry)
        else:
            others.append(entry)

    out = same_vendor + same_type + others
    return out[:400]

# =========================
# UI
# =========================
left, right = st.columns([2, 1])

with left:
    # --- AI Suggest Companions ---
    st.markdown("### 🔮 AI: suggest 2 companions")
    st.caption("Pick one anchor product/variant and let AI suggest two complementary variants to add.")

    anchor_label = st.selectbox("Pick anchor variant", options=["—"] + labels, index=0)
    col_suggest, col_add_suggested = st.columns([1,1])

    with col_suggest:
        disabled = (anchor_label == "—")
        if st.button("✨ Suggest 2 companions", use_container_width=True, disabled=disabled):
            anchor_vid = id_by_label[anchor_label]
            anchor_entry = catalog["var_index"][anchor_vid]
            candidates = _candidate_pool_for_anchor(anchor_vid, catalog["var_index"])
            try:
                vids = gemini_suggest_companions(anchor_entry, candidates, k=2)
                # Validate selections
                valid = []
                for v in vids:
                    if v in catalog["var_index"] and v != anchor_vid and v not in valid:
                        valid.append(v)
                st.session_state.last_suggested_vids = valid
                if valid:
                    lbls = []
                    for v in valid:
                        p = catalog["var_index"][v]["product"]["title"]
                        t = catalog["var_index"][v]["variant"]["title"]
                        lbls.append(f"{p} — {t} ({v.split('/')[-1]})")
                    st.success("Suggested:\n- " + "\n- ".join(lbls))
                else:
                    st.warning("No valid suggestions returned. Try a different anchor.")
            except Exception as e:
                st.error(f"Suggestion failed: {e}")

    with col_add_suggested:
        suggested = st.session_state.last_suggested_vids
        if st.button("➕ Add suggested to bundle", use_container_width=True, disabled=(len(suggested)==0)):
            for v in suggested:
                if v not in st.session_state.picked_variant_ids:
                    st.session_state.picked_variant_ids.append(v)
            st.success("Added suggested items to your bundle.")

    st.markdown("---")

    # --- Search-as-you-type picker (nonce key) ---
    st.markdown("### Search products & variants")
    q = st.text_input(
        "Type to search by product, variant, or ID",
        help="Filters locally. Try part of a product title, variant option, or the numeric variant ID."
    )

    def _label_matches(label: str, term: str) -> bool:
        t = (term or "").strip().lower()
        return True if not t else t in label.lower()

    filtered_labels = [l for l in labels if _label_matches(l, q)]

    multi_key = f"search_results_multiselect_{st.session_state.search_nonce}"
    search_select = st.multiselect(
        "Search results",
        options=filtered_labels,
        key=multi_key,
        help="Pick one or more from results, then click 'Add to bundle'."
    )

    col_add, col_clear = st.columns([1,1])
    with col_add:
        if st.button("➕ Add to bundle", use_container_width=True, disabled=len(search_select) == 0):
            for lab in search_select:
                vid = id_by_label[lab]
                if vid not in st.session_state.picked_variant_ids:
                    st.session_state.picked_variant_ids.append(vid)
            st.session_state.search_nonce += 1
            _rerun()

    with col_clear:
        if st.button("🗑️ Clear all selected", use_container_width=True, disabled=len(st.session_state.picked_variant_ids) == 0):
            st.session_state.picked_variant_ids = []

    st.markdown("#### Your bundle components")
    if not st.session_state.picked_variant_ids:
        st.info("No components yet. Use AI suggestions or search above and add variants.")
    else:
        remove_ids = []
        for vid in st.session_state.picked_variant_ids:
            p = catalog["var_index"][vid]["product"]
            v = catalog["var_index"][vid]["variant"]
            label = f'{p["title"]} — {v["title"]}  ({vid.split("/")[-1]})'
            c1, c2 = st.columns([6,1])
            c1.write(label)
            if c2.button("Remove", key=f"rm_{vid}"):
                remove_ids.append(vid)
        for vid in remove_ids:
            st.session_state.picked_variant_ids.remove(vid)

    picked_variant_ids = st.session_state.picked_variant_ids

    # --- Gemini chat → draft ---
    st.markdown("### Chat with AI about the bundle")
    st.caption("Ask for title, description, and pricing. The AI returns a structured draft you can tweak.")

    if "draft" not in st.session_state:
        st.session_state.draft = {
            "title": "",
            "description_html": "",
            "price": 0.0,
            "compare_at_price": None,
            "components": [{"variant_id": vid, "quantity": 1} for vid in picked_variant_ids],
        }

    user_msg = st.text_input("Your prompt (e.g., “Make a summer skincare duo at 15% off.”)")
    colA, colB = st.columns([1,1])

    with colA:
        if st.button("💡 Propose details with Gemini", use_container_width=True, disabled=not picked_variant_ids):
            context = {
                "selected_variants": [
                    {
                        "variant_id": vid,
                        "title": catalog["var_index"][vid]["variant"]["title"],
                        "product_title": catalog["var_index"][vid]["product"]["title"],
                        "vendor": catalog["var_index"][vid]["product"].get("vendor"),
                        "productType": catalog["var_index"][vid]["product"].get("productType"),
                        "tags": catalog["var_index"][vid]["product"].get("tags"),
                        "price": catalog["var_index"][vid]["variant"].get("price"),
                    } for vid in picked_variant_ids
                ],
                "notes": user_msg or "",
            }
            try:
                bd = gemini_propose_bundle_draft(context)
                draft = bd.dict()

                # ---- Server-side sanity checks ----
                draft["price"] = max(0.0, float(draft.get("price") or 0.0))
                cap = draft.get("compare_at_price")
                draft["compare_at_price"] = float(cap) if cap not in (None, "") else None
                for c in draft.get("components", []):
                    c["quantity"] = max(1, int(c.get("quantity") or 1))

                # Ensure components exist
                if not draft.get("components"):
                    draft["components"] = [{"variant_id": vid, "quantity": 1} for vid in picked_variant_ids]

                st.session_state.draft = draft
                st.success("Draft received from AI.")
            except Exception as e:
                st.error(f"Gemini failed to propose draft: {e}")

    with colB:
        if st.button("🧹 Clear draft", use_container_width=True):
            st.session_state.draft = {
                "title": "",
                "description_html": "",
                "price": 0.0,
                "compare_at_price": None,
                "components": [{"variant_id": vid, "quantity": 1} for vid in picked_variant_ids],
            }

    # Editable draft
    d = st.session_state.draft
    d["title"] = st.text_input("Title", d.get("title",""))
    d["description_html"] = st.text_area("Description (HTML allowed)", d.get("description_html",""), height=160)
    d["price"] = st.number_input("Bundle price", value=float(d.get("price") or 0.0), step=0.5, min_value=0.0)
    d["compare_at_price"] = st.number_input("Compare-at price (optional)", value=float(d.get("compare_at_price") or 0.0), step=0.5, min_value=0.0)

    # Keep components aligned with current picks unless user customized quantities already
    existing_ids = {c.get("variant_id") for c in d.get("components", []) if c.get("variant_id")}
    for vid in picked_variant_ids:
        if vid not in existing_ids:
            d.setdefault("components", []).append({"variant_id": vid, "quantity": 1})
    d["components"] = [c for c in d.get("components", []) if c.get("variant_id") in picked_variant_ids]
    st.session_state.draft = d

with right:
    st.subheader("Create bundle")
    st.caption("Creates a componentized product, sets description & price, activates it, and publishes.")
    if st.button("🚀 Create bundle on Shopify", type="primary", use_container_width=True, disabled=len(st.session_state.picked_variant_ids) == 0):
        try:
            draft: Dict = st.session_state["draft"]

            comp_inputs = build_component_inputs(
                [c["variant_id"] for c in draft.get("components", []) if c.get("variant_id")],
                catalog["var_index"]
            )

            # 1) Create bundle (returns operation id)
            op_id = product_bundle_create(draft["title"], comp_inputs)

            # 2) Poll until created, get product id
            product_id = product_operation_poll(op_id)

            # 3) Update description & set ACTIVE
            product_update(product_id, draft.get("description_html",""), status="ACTIVE")

            # 4) Set price on default variant (requires productId)
            default_vid = get_default_variant_id(product_id)
            price = float(draft.get("price") or 0.0)
            cap = draft.get("compare_at_price")
            cap = float(cap) if cap not in (None, "") else None
            set_price_on_variant(product_id, default_vid, price, cap)

            # 5) Publish to current channel
            publish_to_current_channel(product_id)

            st.success("Bundle created & published!")
            admin_link = f"https://{SHOP}/admin/products/{product_id.split('/')[-1]}"
            st.markdown(f"[Open in Shopify Admin]({admin_link})")
        except Exception as e:
            st.error(f"Failed to create bundle: {e}")
